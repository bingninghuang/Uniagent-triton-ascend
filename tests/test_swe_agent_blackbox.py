"""Verification tests for the blackbox SWE-agent recipe.

Organized by tier following docs/VERIFICATION.md:
  Tier 1: Unit modules (no GPU)
  Tier 2: Dataset loading (no GPU)
  Tier 3: Inference end-to-end (GPU)
  Tier 4: Training end-to-end (GPU, multi-node)
"""

import json
import os
import tempfile

import pytest


# =====================================================================
# Tier 1 — Unit modules (VERIFICATION §1.1–1.6, no GPU)
# =====================================================================


class TestModuleImports:
    """VERIFICATION #1–7: module imports and FQN loading."""

    def test_framework_import(self):  # #1
        from examples.swe_agent_blackbox.framework import SWEAgentFramework
        assert SWEAgentFramework is not None

    def test_agent_runner_import(self):  # #2
        from examples.swe_agent_blackbox.agent_runner import swe_agent_runner, load_agent_config
        assert swe_agent_runner is not None
        assert load_agent_config is not None

    def test_mini_swe_agent_runner_import(self):  # #3
        from examples.swe_agent_blackbox.mini_swe_agent_runner import mini_swe_agent_runner
        assert mini_swe_agent_runner is not None

    def test_reward_import(self):  # #4
        from examples.swe_agent_blackbox.reward import compute_score, evaluate_in_env
        assert compute_score is not None
        assert evaluate_in_env is not None

    def test_parallel_infer_import(self):  # #5
        import examples.swe_agent_blackbox.parallel_infer

    def test_training_fqn_dynamic_load(self):  # #6
        from examples.swe_agent_blackbox.framework import SWEAgentFramework
        from examples.swe_agent_blackbox.agent_runner import swe_agent_runner

    def test_reward_fqn_load(self):  # #7
        from examples.swe_agent_blackbox.reward import compute_score


class TestBuildRewardContext:
    """build_reward_context extracted helper (was §1.6 dedup)."""

    def test_extracts_metadata(self):
        from examples.swe_agent_blackbox.reward import build_reward_context

        tools_kwargs = {
            "reward": {
                "name": "swe_bench",
                "metadata": {"instance_id": "test__repo-123"},
            }
        }
        metadata, eval_timeout = build_reward_context(tools_kwargs)
        assert metadata["data_source"] == "swe_bench"
        assert metadata["reward_model"]["instance_id"] == "test__repo-123"
        assert eval_timeout == 600

    def test_defaults_unknown(self):
        from examples.swe_agent_blackbox.reward import build_reward_context

        metadata, _ = build_reward_context({})
        assert metadata["data_source"] == "unknown"
        assert metadata["reward_model"] == {}

    def test_env_override(self):
        from examples.swe_agent_blackbox.reward import build_reward_context

        os.environ["SWE_AGENT_EVAL_TIMEOUT"] = "300"
        try:
            _, eval_timeout = build_reward_context({})
            assert eval_timeout == 300
        finally:
            del os.environ["SWE_AGENT_EVAL_TIMEOUT"]


class TestComputeScore:
    """VERIFICATION #15: compute_score reads reward_score from extra_info."""

    def test_returns_score_from_extra_info(self):
        from examples.swe_agent_blackbox.reward import compute_score

        assert compute_score("swe_bench", "", "", extra_info={"reward_score": 1.0}) == {"score": 1.0}
        assert compute_score("swe_bench", "", "", extra_info={"reward_score": 0.0}) == {"score": 0.0}

    def test_returns_zero_when_no_extra_info(self):
        from examples.swe_agent_blackbox.reward import compute_score

        assert compute_score("swe_bench", "", "") == {"score": 0.0}
        assert compute_score("swe_bench", "", "", extra_info=None) == {"score": 0.0}

    def test_returns_zero_when_no_reward_score_key(self):
        from examples.swe_agent_blackbox.reward import compute_score

        assert compute_score("swe_bench", "", "", extra_info={"other": 1.0}) == {"score": 0.0}


class TestRewardSpecRegistry:
    """VERIFICATION #13: reward spec registry key matching."""

    @pytest.mark.parametrize("data_source", ["swe_bench", "swe_rebench", "r2e_gym"])
    def test_spec_loadable(self, data_source):
        from examples.swe_agent_blackbox.reward import _get_reward_spec

        cls = _get_reward_spec(data_source)
        assert cls is not None

    def test_unknown_data_source_raises(self):
        from examples.swe_agent_blackbox.reward import _get_reward_spec

        with pytest.raises(ValueError, match="Unknown data_source"):
            _get_reward_spec("nonexistent_benchmark")


class TestRewardInfoInjection:
    """VERIFICATION #16: reward_info → extra_info injection chain."""

    def test_injects_reward_info(self):
        from uni_agent.trainer.framework.types import Trajectory

        traj = [Trajectory(
            prompt_ids=[], response_ids=[], response_mask=[],
            reward_info={"reward_score": 1.0, "resolved": True},
        )]
        fields = {"extra_info": {"data_source": "swe_bench"}}

        # Simulate _score_trajectories merge logic
        reward_info = traj[-1].reward_info
        extra_info = dict(fields.get("extra_info") or {})
        merged = {**extra_info, **reward_info}
        assert merged["reward_score"] == 1.0
        assert merged["resolved"] is True
        assert merged["data_source"] == "swe_bench"

    def test_no_reward_info_no_crash(self):
        from uni_agent.trainer.framework.types import Trajectory

        traj = [Trajectory(prompt_ids=[], response_ids=[], response_mask=[], reward_info={})]
        assert not traj[-1].reward_info


class TestLoadAgentConfig:
    """Agent config YAML loading (VERIFICATION #19)."""

    def test_loads_yaml_list(self):
        from examples.swe_agent_blackbox.agent_runner import load_agent_config

        cfg = load_agent_config("examples/swe_agent_blackbox/config/agent_config.yaml")
        assert isinstance(cfg, dict)
        assert cfg["name"] == "swe_agent"
        assert "interaction" in cfg
        assert "env" in cfg
        assert "tools" in cfg

    def test_loads_yaml_dict(self):
        from examples.swe_agent_blackbox.agent_runner import load_agent_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\n")
            f.flush()
            cfg = load_agent_config(f.name)
        os.unlink(f.name)
        assert cfg == {"key": "value"}

    def test_returns_empty_for_empty_file(self):
        from examples.swe_agent_blackbox.agent_runner import load_agent_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            cfg = load_agent_config(f.name)
        os.unlink(f.name)
        assert cfg == {}


class TestImageRemap:
    """VERIFICATION #28: image name remapping."""

    def test_remote_registry_stripped(self):
        from examples.swe_agent_blackbox.parallel_infer import _remap_image_to_local

        assert _remap_image_to_local("registry.example.com/sweb.eval.x86_64.astropy__astropy-12907:v1") == "sweb.eval.x86_64.astropy__astropy-12907:latest"

    def test_local_image_tag_replaced(self):
        from examples.swe_agent_blackbox.parallel_infer import _remap_image_to_local

        assert _remap_image_to_local("sweb.eval.x86_64.foo__bar:v2") == "sweb.eval.x86_64.foo__bar:latest"

    def test_1776_replacement(self):
        from examples.swe_agent_blackbox.parallel_infer import _remap_image_to_local

        assert _remap_image_to_local("sweb.eval.x86_64.repo_1776_id:tag") == "sweb.eval.x86_64.repo__id:latest"

    def test_no_registry_prefix(self):
        from examples.swe_agent_blackbox.parallel_infer import _remap_image_to_local

        assert _remap_image_to_local("sweb.eval.x86_64.test:123") == "sweb.eval.x86_64.test:latest"


class TestHydraConfig:
    """VERIFICATION #17–18: Hydra config syntax."""

    def test_swe_agent_blackbox_yaml(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("examples/swe_agent_blackbox/config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="swe_agent_blackbox")
        assert cfg.algorithm.adv_estimator == "grpo"

    def test_parallel_infer_yaml(self):
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("examples/swe_agent_blackbox/config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="parallel_infer")
        assert cfg is not None


class TestFQNReferences:
    """VERIFICATION #20–22: FQN references in training config."""

    def test_agent_framework_rollout_adapter(self):  # #20
        from uni_agent.trainer.framework.entry import AgentFrameworkRolloutAdapter
        assert AgentFrameworkRolloutAdapter is not None

    def test_framework_class_fqn(self):  # #21
        from examples.swe_agent_blackbox.framework import SWEAgentFramework
        assert SWEAgentFramework is not None

    def test_agent_runner_fqn(self):  # #22
        from examples.swe_agent_blackbox.agent_runner import swe_agent_runner
        assert swe_agent_runner is not None


# =====================================================================
# Tier 2 — Dataset loading (VERIFICATION §2.4, no GPU)
# =====================================================================


class TestDatasetLoading:
    """VERIFICATION #42–44: parquet loading + image remapping."""

    def test_swe_bench_sample(self):  # #42 (sample)
        from examples.swe_agent_blackbox.parallel_infer import load_swe_dataset

        samples = load_swe_dataset("/tmp/swe_local_8.parquet")
        assert len(samples) == 8
        for s in samples:
            assert "prompt" in s
            tk = s["extra_info"]["tools_kwargs"]
            assert "env" in tk
            assert "reward" in tk
            assert tk["reward"]["name"] == "swe_bench"
            assert ":latest" in tk["env"]["image"]

    def test_swe_bench_full(self):  # #42 (full)
        from examples.swe_agent_blackbox.parallel_infer import load_swe_dataset

        samples = load_swe_dataset("/home/datasets/swe_bench_verified.parquet", max_samples=3)
        assert len(samples) == 3
        for s in samples:
            tk = s["extra_info"]["tools_kwargs"]
            assert tk["reward"]["name"] == "swe_bench"
            assert ":latest" in tk["env"]["image"]

    def test_r2e_gym_sample(self):  # #44 (sample)
        from examples.swe_agent_blackbox.parallel_infer import load_swe_dataset

        samples = load_swe_dataset("/tmp/r2e_local_8.parquet")
        assert len(samples) == 8
        for s in samples:
            tk = s["extra_info"]["tools_kwargs"]
            assert tk["reward"]["name"] == "r2e_gym"
            assert ":latest" in tk["env"]["image"]

    def test_max_samples_limit(self):
        from examples.swe_agent_blackbox.parallel_infer import load_swe_dataset

        samples = load_swe_dataset("/tmp/swe_local_8.parquet", max_samples=3)
        assert len(samples) == 3

    def test_max_samples_negative_returns_all(self):
        from examples.swe_agent_blackbox.parallel_infer import load_swe_dataset

        samples = load_swe_dataset("/tmp/swe_local_8.parquet", max_samples=-1)
        assert len(samples) == 8


# =====================================================================
# Tier 3 — Inference end-to-end (VERIFICATION §2.2, GPU required)
# =====================================================================


@pytest.mark.skipif(
    not os.path.isdir("/data1/models/Qwen/Qwen3.5-4B"),
    reason="Qwen3.5-4B model not found",
)
class TestInferenceEndToEnd:
    """VERIFICATION #36–38: inference end-to-end with GPU."""

    def test_uniagent_runner_swe_bench(self, tmp_path):  # #36
        """Run 1 sample swe_bench inference with uniagent runner."""
        from examples.swe_agent_blackbox.parallel_infer import run_inference

        model_path = os.environ.get("TEST_MODEL_PATH", "/data1/models/Qwen/Qwen3.5-4B")
        tp = int(os.environ.get("TEST_TENSOR_PARALLEL_SIZE", "2"))
        prompt_length = int(os.environ.get("TEST_PROMPT_LENGTH", "4096"))
        response_length = int(os.environ.get("TEST_RESPONSE_LENGTH", "16384"))

        result = run_inference(
            model_path=model_path,
            data_path="/tmp/swe_local_8.parquet",
            max_samples=1,
            n=1,
            engine="vllm",
            tensor_parallel_size=tp,
            gateway_count=1,
            runner="uniagent",
            prompt_length=prompt_length,
            response_length=response_length,
            agent_config_path="examples/swe_agent_blackbox/config/agent_config.yaml",
        )
        assert result is not None
        assert "per_sample_scores" in result
        assert len(result["per_sample_scores"]) == 1

    def test_uniagent_runner_r2e_gym(self):  # #36 + #41
        """Run 1 sample r2e_gym inference with uniagent runner."""
        from examples.swe_agent_blackbox.parallel_infer import run_inference

        model_path = os.environ.get("TEST_MODEL_PATH", "/data1/models/Qwen/Qwen3.5-4B")
        tp = int(os.environ.get("TEST_TENSOR_PARALLEL_SIZE", "2"))
        prompt_length = int(os.environ.get("TEST_PROMPT_LENGTH", "4096"))
        response_length = int(os.environ.get("TEST_RESPONSE_LENGTH", "16384"))

        result = run_inference(
            model_path=model_path,
            data_path="/tmp/r2e_local_8.parquet",
            max_samples=1,
            n=1,
            engine="vllm",
            tensor_parallel_size=tp,
            gateway_count=1,
            runner="uniagent",
            prompt_length=prompt_length,
            response_length=response_length,
            agent_config_path="examples/swe_agent_blackbox/config/agent_config.yaml",
        )
        assert result is not None
        assert "per_sample_scores" in result

    def test_mini_swe_runner_swe_bench(self):  # #37
        """Run 1 sample swe_bench inference with mini_swe runner."""
        from examples.swe_agent_blackbox.parallel_infer import run_inference

        result = run_inference(
            model_path="/data1/models/Qwen/Qwen3.5-4B",
            data_path="/tmp/swe_local_8.parquet",
            max_samples=1,
            n=1,
            engine="vllm",
            tensor_parallel_size=2,
            gateway_count=1,
            runner="mini_swe",
            prompt_length=4096,
            response_length=16384,
        )
        assert result is not None
        assert "per_sample_scores" in result
