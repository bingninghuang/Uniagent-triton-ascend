"""Anthropic Messages shim backed by a uni-agent/verl OpenAI gateway.

Claude Code talks to Anthropic's ``/v1/messages`` API. The blackbox
uni-agent gateway exposes an OpenAI-compatible ``/v1/chat/completions`` API
and records token-level trajectories there. This shim translates the small
Anthropic subset Claude Code uses into OpenAI chat-completion requests, then
packs OpenAI content/tool calls back into Anthropic message or SSE responses.

It is intentionally lightweight. For production-scale runs you can replace it
with a richer adapter such as slime's AnthropicAdapter, but this file keeps the
Triton example self-contained.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)

_FINAL_MANAGEMENT_TOOL_NAMES = {
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "EnterPlanMode",
    "ExitPlanMode",
    "Workflow",
}


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _log_requests_enabled() -> bool:
    return os.environ.get("TRITON_SHIM_LOG_REQUESTS", "0") not in ("0", "false", "False", "no")


def _verbose_logs_enabled() -> bool:
    return os.environ.get("TRITON_VERBOSE_ROLLOUT_LOGS", "0") not in ("0", "false", "False", "no")


def _flatten(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "thinking":
            parts.append(str(block.get("thinking", "")))
        elif block_type == "tool_result":
            parts.append(_flatten(block.get("content")))
        elif block_type == "image":
            parts.append("[image omitted]")
        elif "text" in block:
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _parse_tool_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
    return value if value is not None else {}


_XML_FUNCTION_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</function>\s*(?:</tool_call>)?",
    re.DOTALL,
)
_XML_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _tool_names(tools: list[dict] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if isinstance(tool, dict) and tool.get("name"):
            names.add(str(tool["name"]))
    return names


def _drop_final_management_tools(blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    text_blocks = [block for block in blocks if block.get("type") == "text" and str(block.get("text", "")).strip()]
    tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
    if not text_blocks or not tool_blocks:
        return blocks, False
    if any(str(block.get("name", "")) not in _FINAL_MANAGEMENT_TOOL_NAMES for block in tool_blocks):
        return blocks, False
    return [block for block in blocks if block.get("type") != "tool_use"], True


def _xml_tool_text_to_blocks(content: str, allowed_tool_names: set[str] | None = None) -> list[dict[str, Any]] | None:
    blocks: list[dict[str, Any]] = []
    cursor = 0
    converted = False

    for match in _XML_FUNCTION_RE.finditer(content):
        function_name = match.group(1)
        if allowed_tool_names and function_name not in allowed_tool_names:
            continue

        prefix = content[cursor : match.start()].strip()
        if prefix:
            blocks.append({"type": "text", "text": prefix})

        function_body = match.group(2)
        params = {
            param_match.group(1): param_match.group(2).strip()
            for param_match in _XML_PARAMETER_RE.finditer(function_body)
        }
        blocks.append(
            {
                "type": "tool_use",
                "id": f"toolu_{secrets.token_hex(8)}",
                "name": function_name,
                "input": params,
            }
        )
        cursor = match.end()
        converted = True

    if not converted:
        return None

    suffix = content[cursor:].strip()
    if suffix:
        blocks.append({"type": "text", "text": suffix})
    return blocks


def _anthropic_tools_to_openai(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema")
                    or tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return converted or None


def _translate_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        translated.append({"role": "system", "content": _flatten(system)})

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            pending_user_text: list[str] = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if pending_user_text:
                        translated.append({"role": "user", "content": "\n".join(pending_user_text)})
                        pending_user_text = []
                    translated.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id") or f"toolu_{secrets.token_hex(8)}",
                            "content": _flatten(block.get("content")),
                        }
                    )
                elif isinstance(block, dict) and block.get("type") == "text":
                    pending_user_text.append(str(block.get("text", "")))
                else:
                    pending_user_text.append(_flatten(block))
            if pending_user_text:
                translated.append({"role": "user", "content": "\n".join(pending_user_text)})
        elif role == "assistant":
            texts: list[str] = []
            reasoning: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    texts.append(str(block.get("text", "")))
                elif block_type == "thinking":
                    reasoning.append(str(block.get("thinking", "")))
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id") or f"toolu_{secrets.token_hex(8)}",
                            "type": "function",
                            "function": {
                                "name": block.get("name", "tool"),
                                "arguments": _json_arguments(block.get("input")),
                            },
                        }
                    )
            out: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if tool_calls:
                out["tool_calls"] = tool_calls
            translated.append(out)
    return translated


def _openai_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _request_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _message_blocks_from_openai(
    openai_response: dict[str, Any],
    *,
    allowed_tool_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
    choices = openai_response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    blocks: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        blocks.append({"type": "thinking", "thinking": str(reasoning)})
    content = message.get("content") or ""

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        if content:
            blocks.append({"type": "text", "text": str(content)})
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{secrets.token_hex(8)}",
                    "name": function.get("name", "tool"),
                    "input": _parse_tool_arguments(function.get("arguments")),
                }
            )
    elif content:
        xml_blocks = _xml_tool_text_to_blocks(str(content), allowed_tool_names=allowed_tool_names)
        if xml_blocks:
            blocks.extend(xml_blocks)
            tool_calls = [block for block in xml_blocks if block.get("type") == "tool_use"]
            if _verbose_logs_enabled():
                logger.info("Converted XML-style tool-call text into %d Anthropic tool_use block(s)", len(tool_calls))
        else:
            blocks.append({"type": "text", "text": str(content)})

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    blocks, dropped_management_tools = _drop_final_management_tools(blocks)
    if dropped_management_tools:
        tool_calls = []
        if _verbose_logs_enabled():
            logger.info("Dropped final task-management tool call(s) after assistant text")

    if tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage_raw = openai_response.get("usage") or {}
    usage = {
        "input_tokens": int(usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0),
        "output_tokens": int(usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0),
    }
    return blocks, stop_reason, usage


def _anthropic_message_response(
    request_body: dict[str, Any],
    blocks: list[dict[str, Any]],
    stop_reason: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": request_body.get("model", "uni-agent-actor"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def _sse_payload(blocks: list[dict[str, Any]], stop_reason: str, usage: dict[str, int]) -> bytes:
    chunks: list[str] = []
    message_start = {
        "type": "message_start",
        "message": {
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "role": "assistant",
            "model": "uni-agent-actor",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
        },
    }
    chunks.append(f"event: message_start\ndata: {json.dumps(message_start, ensure_ascii=False)}\n\n")
    for index, block in enumerate(blocks):
        block_type = block.get("type")
        if block_type == "thinking":
            start_block = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block.get("thinking", "")}
        elif block_type == "tool_use":
            start_block = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
            }
        else:
            start_block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text", "")}
        chunks.append(
            "event: content_block_start\n"
            f"data: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': start_block}, ensure_ascii=False)}\n\n"
        )
        chunks.append(
            "event: content_block_delta\n"
            f"data: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': delta}, ensure_ascii=False)}\n\n"
        )
        chunks.append(
            "event: content_block_stop\n"
            f"data: {json.dumps({'type': 'content_block_stop', 'index': index}, ensure_ascii=False)}\n\n"
        )
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    }
    chunks.append(f"event: message_delta\ndata: {json.dumps(message_delta, ensure_ascii=False)}\n\n")
    chunks.append("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
    return "".join(chunks).encode("utf-8")


class AnthropicOpenAIShim:
    """Background HTTP server translating Anthropic Messages to OpenAI chat."""

    def __init__(
        self,
        *,
        openai_base_url: str,
        openai_api_key: str = "EMPTY",
        model_name: str = "default",
        host: str = "0.0.0.0",
        port: int | None = None,
        request_timeout: float = 600.0,
    ) -> None:
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "AnthropicOpenAIShim":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._server is not None:
            return
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "TritonAnthropicOpenAIShim/0.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("shim: " + fmt, *args)

            def _send_json(self, code: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _send_bytes(self, code: int, payload: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                path = urllib.parse.urlparse(self.path).path
                if path in ("/healthz", "/v1/models"):
                    self._send_json(200, {"ok": True})
                else:
                    self._send_json(404, {"error": f"unknown path: {path}"})

            def do_POST(self) -> None:  # noqa: N802
                path = urllib.parse.urlparse(self.path).path
                if path == "/v1/messages/count_tokens":
                    self._send_json(200, {"input_tokens": 0})
                    return
                if path != "/v1/messages":
                    self._send_json(404, {"error": f"unknown path: {path}"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                    body = json.loads(self.rfile.read(length).decode("utf-8") if length else "{}")
                    payload: dict[str, Any] = {
                        "model": owner.model_name,
                        "messages": _translate_messages(body),
                    }
                    max_tokens = body.get("max_tokens")
                    if max_tokens is not None:
                        payload["max_tokens"] = max_tokens
                    for key in ("temperature", "top_p", "stop"):
                        if key in body:
                            payload[key] = body[key]
                    if "stop_sequences" in body:
                        payload["stop"] = body["stop_sequences"]
                    tools = _anthropic_tools_to_openai(body.get("tools"))
                    if tools:
                        payload["tools"] = tools

                    started_at = time.time()
                    if _log_requests_enabled():
                        logger.info(
                            "shim request: model=%s messages=%s tools=%s max_tokens=%s stream=%s",
                            owner.model_name,
                            len(payload["messages"]),
                            bool(tools),
                            payload.get("max_tokens"),
                            bool(body.get("stream") is True or "text/event-stream" in self.headers.get("Accept", "")),
                        )
                    response = _request_json(
                        _openai_endpoint(owner.openai_base_url),
                        payload,
                        api_key=owner.openai_api_key,
                        timeout=owner.request_timeout,
                    )
                    blocks, stop_reason, usage = _message_blocks_from_openai(
                        response,
                        allowed_tool_names=_tool_names(body.get("tools")),
                    )
                    if _log_requests_enabled():
                        logger.info(
                            "shim response: stop_reason=%s elapsed=%.2fs input_tokens=%s output_tokens=%s",
                            stop_reason,
                            time.time() - started_at,
                            usage.get("input_tokens") if isinstance(usage, dict) else None,
                            usage.get("output_tokens") if isinstance(usage, dict) else None,
                        )
                    if body.get("stream") is True or "text/event-stream" in self.headers.get("Accept", ""):
                        self._send_bytes(200, _sse_payload(blocks, stop_reason, usage), "text/event-stream")
                    else:
                        self._send_json(200, _anthropic_message_response(body, blocks, stop_reason, usage))
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    logger.warning("OpenAI gateway returned HTTP %s: %s", exc.code, detail[:1000])
                    self._send_json(502, {"error": "openai_gateway_error", "status": exc.code, "detail": detail})
                except Exception as exc:  # pragma: no cover - defensive server boundary
                    logger.exception("Anthropic/OpenAI shim request failed")
                    self._send_json(500, {"error": type(exc).__name__, "detail": str(exc)})

        bind_port = self.port or 0
        try:
            self._server = _ReusableThreadingHTTPServer((self.host, bind_port), Handler)
        except OSError as exc:
            if bind_port and exc.errno in (errno.EADDRINUSE, 98):
                logger.warning("Shim port %s is already in use; falling back to an ephemeral port", bind_port)
                self._server = _ReusableThreadingHTTPServer((self.host, 0), Handler)
            else:
                raise
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"triton-anthropic-openai-shim-{self.port}",
            daemon=True,
        )
        self._thread.start()
        if _verbose_logs_enabled():
            logger.info("Started Anthropic/OpenAI shim on %s:%s -> %s", self.host, self.port, self.openai_base_url)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
