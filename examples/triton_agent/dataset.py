"""Dataset adapter for Triton Claude Code blackbox training."""

from __future__ import annotations

try:
    from verl.utils.dataset.rl_dataset import RLHFDataset
except ImportError:  # pragma: no cover - imported only inside verl training
    RLHFDataset = object  # type: ignore[assignment]


class TritonKernelBenchDataset(RLHFDataset):  # type: ignore[misc]
    """Ensure verl-standard reward fields exist for blackbox rows."""

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        extra_info = row_dict.get("extra_info", {})
        tools_kwargs = extra_info.get("tools_kwargs", {}) if isinstance(extra_info, dict) else {}
        reward_config = tools_kwargs.get("reward", {}) if isinstance(tools_kwargs, dict) else {}
        metadata = reward_config.get("metadata", {}) if isinstance(reward_config, dict) else {}

        row_dict.setdefault("data_source", reward_config.get("name", "triton_kernelbench"))
        row_dict.setdefault("reward_model", {"ground_truth": metadata, "style": "rule"})
        return row_dict
