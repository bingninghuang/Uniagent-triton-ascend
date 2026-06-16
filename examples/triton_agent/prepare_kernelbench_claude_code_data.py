"""Prepare KernelBench data for Triton Claude Code blackbox training."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import Dataset

SYSTEM_PROMPT = "You are a coding agent solving Triton Ascend KernelBench tasks."
_DEFAULT_LEVELS = ("level_1", "level_2")
_WARMUP_EXCLUDE_KEYWORDS = (
    "conv_transpose",
    "conv_transposed",
    "transpose3d",
    "transposed_3d",
    "conv3d",
    "3d_convolution",
    "attention",
    "transformer",
    "conv2d",
    "conv_standard",
)

USER_PROMPT = """Implement the Triton Ascend operator described by the task workspace.

The blackbox runner starts Claude Code in a sandbox containing CLAUDE.md,
INSTRUCTIONS.md, local `.claude/skills`, local `.claude/refs`, and
src/{op_name}.py. The task is already extracted and staged; do not invoke
triton-task-extractor. Implement ModelNew and validate with the compact
commands in INSTRUCTIONS.md plus the local PR205 triton-op-verifier scripts.
Use OPERATOR_PYTHON through `bash tools/run_npu_command.sh` for NPU verification
when it is set. Do not use src directly as verify_dir; stage files and
src/*.json sidecars into output/verify first. AST validation alone is not
success. Only verify_result.json with passed_cases == total_cases is success.
Any nonzero validation command is failure to repair, not success. Use direct
Read/Bash/Edit/Write calls for edits and validation; use local Skill only when
needed to find bundled guidance. Pass tensors directly to Triton kernels; do
not pass tensor.data_ptr(). `Unsupported ptr type ... in
tl.load` is an implementation bug, not an environment issue.
"""


def _safe_name(value: Any, idx: int) -> str:
    raw = str(value or f"kernelbench_{idx}")
    raw = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_")
    if not raw:
        raw = f"kernelbench_{idx}"
    if raw[0].isdigit():
        raw = f"task_{raw}"
    return raw[:96]


def _parse_levels(levels: str | None) -> tuple[str, ...]:
    if not levels:
        return _DEFAULT_LEVELS
    parsed = tuple(part.strip() for part in levels.split(",") if part.strip())
    if len(parsed) == 1 and parsed[0].lower() in {"all", "*"}:
        return tuple(f"level_{idx}" for idx in range(5))
    return parsed or _DEFAULT_LEVELS


def _level_to_int(level: str) -> int | None:
    text = str(level).strip()
    if text.startswith("level_"):
        text = text[len("level_") :]
    elif text.startswith("level"):
        text = text[len("level") :]
    try:
        return int(text)
    except ValueError:
        return None


def _level_dir_names(levels: tuple[str, ...]) -> set[str]:
    names: set[str] = set()
    for level in levels:
        text = str(level).strip()
        if not text:
            continue
        names.add(text)
        names.add(text.replace("level_", "level"))
        value = _level_to_int(text)
        if value is not None:
            names.add(f"level{value}")
            names.add(f"level_{value}")
    return names


def _parse_npukernelbench_filename(path: Path) -> tuple[int | str, str]:
    match = re.match(r"^(\d+)[_-](.+)$", path.stem)
    if match:
        return int(match.group(1)), match.group(2)
    return path.stem, path.stem


def _local_npukernelbench_rows(source: str, levels: tuple[str, ...]) -> list[dict[str, Any]]:
    root = Path(source)
    if not root.is_dir():
        return []
    level_names = _level_dir_names(levels)
    level_dirs = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and (not level_names or child.name in level_names)
    ]
    rows: list[dict[str, Any]] = []
    for level_dir in level_dirs:
        level = _level_to_int(level_dir.name)
        for py_file in sorted(level_dir.glob("*.py")):
            problem_id, name = _parse_npukernelbench_filename(py_file)
            support_files: dict[str, str] = {}
            for sidecar in (py_file.with_suffix(".json"), py_file.with_name(f"{py_file.stem}_all_case.json")):
                if sidecar.is_file():
                    support_files[sidecar.name] = sidecar.read_text(encoding="utf-8", errors="replace")
            rows.append(
                {
                    "code": py_file.read_text(encoding="utf-8", errors="replace"),
                    "level": level if level is not None else level_dir.name,
                    "problem_id": problem_id,
                    "name": name,
                    "support_files": support_files,
                }
            )
    return rows


def _local_kernelbench_files(source: str, levels: tuple[str, ...]) -> list[str]:
    path = Path(source)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        return []
    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        return []
    level_names = set(levels)
    matched = [
        file
        for file in parquet_files
        if file.stem in level_names or file.name in {f"{level}.parquet" for level in level_names}
    ]
    return [str(file) for file in (matched or parquet_files)]


def _filter_by_level_if_present(ds: Dataset, levels: tuple[str, ...]) -> Dataset:
    target_levels = {value for value in (_level_to_int(level) for level in levels) if value is not None}
    if not target_levels or "level" not in getattr(ds, "column_names", []):
        return ds
    return ds.filter(lambda row: int(row.get("level", -1)) in target_levels)


def _load_kernelbench_levels(hf_dataset: str, levels: tuple[str, ...]) -> Dataset:
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets, load_dataset

    local_rows = _local_npukernelbench_rows(hf_dataset, levels)
    if local_rows:
        return HFDataset.from_list(local_rows)

    local_files = _local_kernelbench_files(hf_dataset, levels)
    if local_files:
        ds = load_dataset("parquet", data_files=local_files, split="train")
        return _filter_by_level_if_present(ds, levels)

    loaded = []
    direct_load_failed = False
    for level in levels:
        try:
            loaded.append(load_dataset(hf_dataset, split=level))
        except ValueError as exc:
            if "Unknown split" not in str(exc) and "unknown split" not in str(exc):
                raise
            direct_load_failed = True
            break
    if loaded and not direct_load_failed:
        return loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)
    ds = load_dataset(hf_dataset, split="train")
    return _filter_by_level_if_present(ds, levels)


def _parse_keywords(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def _eval_int_expr(node: ast.AST, names: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_int_expr(node.operand, names)
        return -value if value is not None else None
    if isinstance(node, ast.BinOp):
        left = _eval_int_expr(node.left, names)
        right = _eval_int_expr(node.right, names)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right:
            return left // right
    return None


def _shape_numel(shape: tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    total = 1
    for value in shape:
        total *= value
    return total


def _torch_factory_shape(node: ast.AST, names: dict[str, int]) -> tuple[int, ...] | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in {"rand", "randn", "empty", "zeros", "ones"}:
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "torch":
        return None
    shape = tuple(_eval_int_expr(arg, names) for arg in node.args)
    if any(value is None for value in shape):
        return None
    return tuple(int(value) for value in shape if value is not None)


def _attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _matmul_output_shape(node: ast.AST, input_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...] | None:
    left: ast.AST | None = None
    right: ast.AST | None = None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        left, right = node.left, node.right
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "matmul"
            and len(node.args) >= 2
        ):
            left, right = node.args[0], node.args[1]
    if left is None or right is None:
        return None

    left_shape = input_shapes.get(_attr_name(left) or "")
    if isinstance(right, ast.Attribute) and right.attr == "T":
        base_shape = input_shapes.get(_attr_name(right.value) or "")
        right_shape = tuple(reversed(base_shape)) if base_shape else None
    else:
        right_shape = input_shapes.get(_attr_name(right) or "")

    if not left_shape or not right_shape or len(left_shape) != 2 or len(right_shape) != 2:
        return None
    return (left_shape[0], right_shape[1])


def _infer_kernelbench_shape_limits(code: str) -> tuple[int | None, int | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None

    names: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _eval_int_expr(node.value, names)
            if value is not None:
                names[node.targets[0].id] = value

    input_shapes: dict[str, tuple[int, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    shape = _torch_factory_shape(stmt.value, names)
                    if shape:
                        input_shapes[stmt.targets[0].id] = shape

    max_input = max((_shape_numel(shape) or 0 for shape in input_shapes.values()), default=0) or None

    output_numel: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return):
                    shape = _matmul_output_shape(stmt.value, input_shapes)
                    output_numel = _shape_numel(shape)
                    if output_numel is not None:
                        break
    return max_input, output_numel


def _row_search_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("name", "")),
            str(row.get("problem_id", "")),
            str(row.get("level", "")),
            str(row.get("code", row.get("reference_code", ""))),
        ]
    ).lower()


def _apply_sample_filter(ds: Dataset) -> Dataset:
    mode = os.environ.get("TRITON_KERNELBENCH_FILTER_MODE", "warmup").strip().lower()
    include = _parse_keywords(os.environ.get("TRITON_KERNELBENCH_INCLUDE_KEYWORDS"))
    exclude_env = os.environ.get("TRITON_KERNELBENCH_EXCLUDE_KEYWORDS")
    if exclude_env is not None:
        exclude = _parse_keywords(exclude_env)
    elif mode in ("all", "none", "off", "false", "0"):
        exclude = ()
    elif mode == "warmup":
        exclude = _WARMUP_EXCLUDE_KEYWORDS
    else:
        exclude = ()
    max_code_chars = _optional_int_env("TRITON_KERNELBENCH_MAX_CODE_CHARS")
    max_input_elements = _optional_int_env("TRITON_KERNELBENCH_MAX_INPUT_ELEMENTS")
    max_output_elements = _optional_int_env("TRITON_KERNELBENCH_MAX_OUTPUT_ELEMENTS")

    if (
        not include
        and not exclude
        and max_code_chars is None
        and max_input_elements is None
        and max_output_elements is None
    ):
        print("[kernelbench] sample filter disabled")
        return ds

    def keep(row: dict[str, Any]) -> bool:
        code = row.get("code") or row.get("reference_code") or ""
        if max_code_chars is not None and len(str(code)) > max_code_chars:
            return False
        if max_input_elements is not None or max_output_elements is not None:
            input_elements, output_elements = _infer_kernelbench_shape_limits(str(code))
            if max_input_elements is not None and input_elements is not None and input_elements > max_input_elements:
                return False
            if max_output_elements is not None and output_elements is not None and output_elements > max_output_elements:
                return False
        text = _row_search_text(row)
        if include and not any(keyword in text for keyword in include):
            return False
        if exclude and any(keyword in text for keyword in exclude):
            return False
        return True

    before = len(ds)
    filtered = ds.filter(keep)
    print(
        "[kernelbench] sample filter: "
        f"mode={mode}, before={before}, after={len(filtered)}, "
        f"include={include or 'ALL'}, exclude={exclude or 'NONE'}, "
        f"max_code_chars={max_code_chars or 'NONE'}, "
        f"max_input_elements={max_input_elements or 'NONE'}, "
        f"max_output_elements={max_output_elements or 'NONE'}"
    )
    return filtered


def _row_to_kernelbench_record(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    task_code = row.get("code") or row.get("reference_code") or ""
    if not task_code:
        return None
    level = row.get("level", "")
    problem_id = row.get("problem_id", idx)
    name = row.get("name", f"problem_{problem_id}")
    op_name = _safe_name(f"kernelbench_l{level}_{problem_id}_{name}", idx)
    instruction = (
        f"Implement KernelBench problem {problem_id}"
        f"{f' ({name})' if name else ''} as an Ascend NPU Triton operator."
    )
    extra_info = {
        "instruction": instruction,
        "scenario": "npu_operator",
        "op_name": op_name,
        "arch": os.environ.get("TRITON_KERNELBENCH_ARCH", "ascend910b1"),
        "task_code": task_code,
        "operator_backend": os.environ.get("TRITON_KERNELBENCH_OPERATOR_BACKEND", "triton"),
        "entry_point": "Model",
        "uid": op_name,
        "data_source": "kernelbench",
        "ability": "kernel_optimization",
        "kernelbench_level": str(level),
        "kernelbench_problem_id": str(problem_id),
        "kernelbench_name": str(name),
    }
    support_files = row.get("support_files")
    if isinstance(support_files, dict) and support_files:
        extra_info["support_files"] = support_files
    return {"extra_info": extra_info}


def _normalize_remote_host(host: str) -> str:
    host = str(host).strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    parsed = urlparse(host)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid remote sandbox host: {host!r}")
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    return f"{parsed.scheme}://{hostname}"


def _remote_endpoint_from_url(text: str, auth_token: str | None = None) -> dict[str, Any]:
    raw = text.strip()
    token = auth_token
    if token is None:
        for sep in ("|", "="):
            if sep in raw:
                raw, token = raw.rsplit(sep, 1)
                break
    if token is None:
        raw, port, token = raw.rsplit(":", 2)
        raw = f"{raw}:{port}"
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if parsed.port is None:
        raise ValueError(f"Remote sandbox endpoint must include a port: {text!r}")
    return {
        "type": "local_attach",
        "host": _normalize_remote_host(f"{parsed.scheme}://{parsed.hostname}"),
        "port": int(parsed.port),
        "auth_token": str(token),
    }


def _coerce_remote_endpoint(endpoint: Any) -> dict[str, Any]:
    if isinstance(endpoint, str):
        return _remote_endpoint_from_url(endpoint)
    if not isinstance(endpoint, dict):
        raise TypeError(f"Remote sandbox endpoint must be dict or str, got {type(endpoint)!r}")
    if "url" in endpoint:
        result = _remote_endpoint_from_url(str(endpoint["url"]), endpoint.get("auth_token"))
    else:
        result = {
            "type": "local_attach",
            "host": _normalize_remote_host(str(endpoint["host"])),
            "port": int(endpoint["port"]),
            "auth_token": str(endpoint["auth_token"]),
        }
    for key in ("timeout", "startup_timeout", "proxy"):
        if key in endpoint and endpoint[key] not in (None, ""):
            result[key] = endpoint[key]
    return result


def _remote_sandbox_pool_from_env() -> list[dict[str, Any]]:
    raw_pool = os.environ.get("TRITON_REMOTE_SANDBOX_POOL_JSON") or os.environ.get("TRITON_REMOTE_SANDBOX_POOL")
    if raw_pool:
        try:
            loaded = json.loads(raw_pool)
            if isinstance(loaded, dict):
                loaded = loaded.get("endpoints", [loaded])
            return [_coerce_remote_endpoint(item) for item in loaded]
        except json.JSONDecodeError:
            entries = [entry.strip() for entry in raw_pool.replace("\n", ",").split(",") if entry.strip()]
            return [_coerce_remote_endpoint(entry) for entry in entries]

    host = os.environ.get("TRITON_REMOTE_SANDBOX_HOST")
    ports = os.environ.get("TRITON_REMOTE_SANDBOX_PORTS")
    if not host or not ports:
        return []
    tokens_raw = os.environ.get("TRITON_REMOTE_SANDBOX_AUTH_TOKENS") or os.environ.get("TRITON_REMOTE_SANDBOX_AUTH_TOKEN")
    if not tokens_raw:
        raise ValueError("TRITON_REMOTE_SANDBOX_AUTH_TOKEN(S) is required when TRITON_REMOTE_SANDBOX_PORTS is set")
    port_list = [int(port.strip()) for port in ports.split(",") if port.strip()]
    token_list = [token.strip() for token in tokens_raw.split(",") if token.strip()]
    if len(token_list) == 1:
        token_list = token_list * len(port_list)
    if len(port_list) != len(token_list):
        raise ValueError("TRITON_REMOTE_SANDBOX_PORTS and TRITON_REMOTE_SANDBOX_AUTH_TOKENS lengths differ")
    return [
        {
            "type": "local_attach",
            "host": _normalize_remote_host(host),
            "port": port,
            "auth_token": token,
        }
        for port, token in zip(port_list, token_list, strict=True)
    ]


def _env_variables() -> dict[str, str]:
    env = {
        "PIP_PROGRESS_BAR": "off",
        "PIP_CACHE_DIR": "~/.cache/pip",
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "TQDM_DISABLE": "1",
        "GIT_PAGER": "cat",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "WORKSPACE_BASE": "/opt/workspace/agent_workdir",
        "TRITON_PIPELINE_ERROR_PREVIEW_CHARS": os.environ.get("TRITON_PIPELINE_ERROR_PREVIEW_CHARS", "2000"),
    }
    optional_eval_env = {
        "TRITON_EVAL_LOCK_DIR": "EVAL_LOCK_DIR",
        "TRITON_EVAL_DEVICE_PREFIX": "EVAL_DEVICE_PREFIX",
        "TRITON_EVAL_DEVICE_COUNT": "EVAL_DEVICE_COUNT",
        "TRITON_EVAL_DEVICE_IDS": "EVAL_DEVICE_IDS",
        "TRITON_EVAL_ENV_NAME": "EVAL_ENV_NAME",
        "TRITON_EVAL_RETRY_INTERVAL": "EVAL_RETRY_INTERVAL",
        "TRITON_EVAL_TIMEOUT": "EVAL_TIMEOUT",
        "TRITON_EVAL_VERBOSE": "EVAL_VERBOSE",
    }
    for source, target in optional_eval_env.items():
        if os.environ.get(source):
            env[target] = os.environ[source]
    return env


def _local_deployment_config() -> dict[str, Any]:
    image = os.environ.get("TRITON_CLAUDE_IMAGE", "triton-claude-code-env:latest")
    deployment = {
        "type": "local",
        "image": image,
        "command": "exec python3 -m swerex.server --host 0.0.0.0 --port {port} --auth-token {token}",
        "timeout": float(os.environ.get("TRITON_SANDBOX_TIMEOUT", "600")),
        "startup_timeout": float(os.environ.get("TRITON_SANDBOX_STARTUP_TIMEOUT", "600")),
        "container_runtime": os.environ.get("TRITON_CONTAINER_RUNTIME", "docker"),
        "extra_run_args": [
            "--ipc=host",
            "--privileged",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-v",
            "/dev:/dev",
            "-v",
            "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro",
            "-v",
            "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro",
            "-v",
            "/usr/local/dcmi:/usr/local/dcmi:ro",
            "-v",
            "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro",
            "-v",
            "/etc/ascend_install.info:/etc/ascend_install.info:ro",
            "-v",
            "/tmp/shared_npu_lock:/shared/device-locks",
        ],
    }
    network = os.environ.get("TRITON_CONTAINER_NETWORK", "").strip()
    if network:
        deployment["network"] = network
    return deployment


def _default_env_config() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    remote_pool = _remote_sandbox_pool_from_env()
    deployment_type = os.environ.get("TRITON_SANDBOX_DEPLOYMENT", "").strip()
    if remote_pool and not deployment_type:
        deployment_type = "local_attach"

    if deployment_type == "local_attach":
        if remote_pool:
            deployment = dict(remote_pool[0])
        else:
            deployment = {
                "type": "local_attach",
                "host": _normalize_remote_host(os.environ["TRITON_REMOTE_SANDBOX_HOST"]),
                "port": int(os.environ["TRITON_REMOTE_SANDBOX_PORT"]),
                "auth_token": os.environ["TRITON_REMOTE_SANDBOX_AUTH_TOKEN"],
            }
            remote_pool = [dict(deployment)]
        deployment.setdefault("timeout", float(os.environ.get("TRITON_SANDBOX_TIMEOUT", "600")))
        deployment.setdefault("startup_timeout", float(os.environ.get("TRITON_SANDBOX_STARTUP_TIMEOUT", "600")))
    else:
        deployment = _local_deployment_config()

    return {"deployment": deployment, "env_variables": _env_variables()}, remote_pool


def _row_to_blackbox_record(
    row: dict[str, Any],
    idx: int,
    env_config: dict[str, Any],
    remote_sandbox_pool: list[dict[str, Any]],
) -> dict[str, Any] | None:
    kernelbench_record = _row_to_kernelbench_record(row, idx)
    if kernelbench_record is None:
        return None
    metadata = dict(kernelbench_record["extra_info"])
    metadata["arch"] = os.environ.get("TRITON_KERNELBENCH_ARCH", metadata.get("arch", "ascend910b1"))
    metadata["operator_backend"] = os.environ.get(
        "TRITON_KERNELBENCH_OPERATOR_BACKEND",
        metadata.get("operator_backend", "triton"),
    )
    op_name = metadata.get("op_name", f"operator_{idx}")
    tools_kwargs = {
        "env": env_config,
        "reward": {
            "name": "triton_kernelbench",
            "metadata": metadata,
            "eval_timeout": float(os.environ.get("TRITON_EVAL_TIMEOUT", "600")),
        },
        "claude_code": {
            "time_budget_sec": int(os.environ.get("TRITON_CLAUDE_TIME_BUDGET_SEC", "1800")),
            "prompt": os.environ.get("TRITON_CLAUDE_PROMPT", ""),
            "shim_public_host": os.environ.get("TRITON_SHIM_PUBLIC_HOST", ""),
            "artifact_dir": os.environ.get("TRITON_CLAUDE_ARTIFACT_DIR", ""),
            "remote_sandbox_pool": remote_sandbox_pool,
        },
    }
    if not tools_kwargs["claude_code"]["prompt"]:
        tools_kwargs["claude_code"].pop("prompt")
    if not tools_kwargs["claude_code"]["shim_public_host"]:
        tools_kwargs["claude_code"].pop("shim_public_host")
    if not tools_kwargs["claude_code"]["artifact_dir"]:
        tools_kwargs["claude_code"].pop("artifact_dir")
    if not tools_kwargs["claude_code"]["remote_sandbox_pool"]:
        tools_kwargs["claude_code"].pop("remote_sandbox_pool")

    return {
        "data_source": "triton_kernelbench",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(op_name=op_name)},
        ],
        "reward_model": {"ground_truth": metadata, "style": "rule"},
        "agent_name": "triton_claude_code",
        "extra_info": {
            "index": idx,
            "uid": metadata.get("uid", op_name),
            "tools_kwargs": tools_kwargs,
        },
    }


def build_dataset(
    *,
    levels: tuple[str, ...],
    max_rows: int | None,
    hf_dataset: str,
    env_config: dict[str, Any],
    remote_sandbox_pool: list[dict[str, Any]],
) -> Dataset:
    ds = _load_kernelbench_levels(hf_dataset, levels)
    ds = _apply_sample_filter(ds)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    rows = []
    skipped = 0
    for idx, row in enumerate(ds):
        record = _row_to_blackbox_record(row, idx, env_config, remote_sandbox_pool)
        if record is None:
            skipped += 1
            continue
        rows.append(record)
    if not rows:
        raise ValueError(f"No usable rows found in {hf_dataset!r} levels {levels!r}")
    print(f"Prepared {len(rows)} rows (skipped={skipped}, levels={levels})", flush=True)
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", default="examples/triton_agent/kernelbench_claude_code.parquet")
    parser.add_argument("--levels", default=os.environ.get("TRITON_KERNELBENCH_LEVELS", "level_1"))
    parser.add_argument("--max-rows", type=int, default=int(os.environ.get("TRITON_KERNELBENCH_MAX_ROWS", "128")))
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TRITON_KERNELBENCH_DATASET", "ScalingIntelligence/KernelBench"),
    )
    args = parser.parse_args()

    env_config, remote_sandbox_pool = _default_env_config()
    dataset = build_dataset(
        levels=_parse_levels(args.levels),
        max_rows=args.max_rows,
        hf_dataset=args.dataset,
        env_config=env_config,
        remote_sandbox_pool=remote_sandbox_pool,
    )
    output = Path(args.output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output))
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
