"""SWEBench-specific dataset that injects verl-standard reward fields."""

from verl.utils.dataset.rl_dataset import RLHFDataset


class SWEBenchDataset(RLHFDataset):

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        extra_info = row_dict.get("extra_info", {})
        tools_kwargs = extra_info.get("tools_kwargs", {})
        reward_config = tools_kwargs.get("reward", {})

        row_dict.setdefault("data_source", reward_config.get("name", "unknown"))
        row_dict.setdefault("reward_model", {"ground_truth": {}})

        return row_dict
