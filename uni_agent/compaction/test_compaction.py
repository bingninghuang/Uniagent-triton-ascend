"""Tests for the conversation compaction service (uni_agent.compaction).

Covers:
* TokenMeter heuristic accuracy and usage-priority behavior.
* CompactionConfig.from_env parsing (defaults / overrides / invalid input).
* Trigger logic (threshold, min messages, enabled, failure backoff).
* LLM-summary compaction: verbatim system/task preservation, instruction
  placement, rejection and error fallbacks, multi-compaction (checkpoint
  merge setup).
* Segment workspace snapshots (impl file + verify result collection).
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uni_agent.compaction import (  # noqa: E402
    CHECKPOINT_PREAMBLE,
    COMPACTION_INSTRUCTION,
    CompactionConfig,
    ConversationCompactor,
    TokenMeter,
    estimate_message,
    estimate_text,
    frame_summary,
)
from uni_agent.compaction.compactor import _extract_compact_result  # noqa: E402


# ---------------------------------------------------------------------------
# TokenMeter
# ---------------------------------------------------------------------------


class TestTokenMeter(unittest.TestCase):
    def test_estimate_text_empty(self):
        self.assertEqual(estimate_text(""), 0)

    def test_estimate_text_density(self):
        # 4 chars/token + 4 block overhead: "abcd" -> 1 + 4 = 5
        self.assertEqual(estimate_text("abcd"), 5)
        # 5 chars -> 2 tokens -> 6; 8 chars -> 2 + 4 = 6
        self.assertEqual(estimate_text("abcde"), 6)
        self.assertEqual(estimate_text("abcdefgh"), 6)

    def test_estimate_message_role_overhead(self):
        # content 4 chars (5 tokens) + role overhead 4 -> 9
        msg = {"role": "user", "content": "abcd"}
        self.assertEqual(estimate_message(msg), 9)

    def test_estimate_message_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "abcd", "arguments": '{"a": 1}'},
                }
            ],
        }
        # content 0; tool call: name 4 chars (1 tok) + args 8 chars (2 tok)
        # + block overhead 4 = 7; + role overhead 4 = 11
        self.assertEqual(estimate_message(msg), 11)

    def test_estimate_messages_sums(self):
        messages = [
            {"role": "system", "content": "abcd"},
            {"role": "user", "content": "abcdefgh"},
        ]
        meter = TokenMeter()
        self.assertEqual(
            meter.estimate_messages(messages),
            estimate_message(messages[0]) + estimate_message(messages[1]),
        )

    def test_current_tokens_prefers_usage(self):
        meter = TokenMeter()
        messages = [{"role": "user", "content": "x" * 400}]  # ~104 est tokens
        # provider usage wins
        self.assertEqual(meter.current_tokens(messages, (1000, 50)), 1050)
        # missing usage falls back to heuristic
        self.assertEqual(
            meter.current_tokens(messages, (None, None)),
            meter.estimate_messages(messages),
        )
        self.assertEqual(
            meter.current_tokens(messages), meter.estimate_messages(messages)
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestCompactionConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = CompactionConfig.from_env({})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.threshold_ratio, 0.8)
        self.assertEqual(cfg.max_context_tokens, 40960)
        self.assertEqual(cfg.threshold_tokens, int(40960 * 0.8))

    def test_env_overrides(self):
        cfg = CompactionConfig.from_env(
            {
                "TRITON_COMPACTION_ENABLED": "0",
                "TRITON_COMPACTION_THRESHOLD": "0.5",
                "TRITON_COMPACTION_MAX_CONTEXT": "8192",
                "TRITON_COMPACTION_BACKOFF": "5",
            }
        )
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.threshold_ratio, 0.5)
        self.assertEqual(cfg.max_context_tokens, 8192)
        self.assertEqual(cfg.failure_backoff_steps, 5)
        self.assertEqual(cfg.threshold_tokens, 4096)

    def test_invalid_env_falls_back(self):
        cfg = CompactionConfig.from_env(
            {
                "TRITON_COMPACTION_THRESHOLD": "not-a-float",
                "TRITON_COMPACTION_MAX_CONTEXT": "not-an-int",
                "TRITON_COMPACTION_ENABLED": "bogus",
            }
        )
        self.assertEqual(cfg.threshold_ratio, 0.8)
        self.assertEqual(cfg.max_context_tokens, 40960)
        self.assertFalse(cfg.enabled)  # anything not in {1,true,yes,on} -> False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_messages(num_rounds: int = 4, round_chars: int = 2000):
    """A synthetic [system, user task, (assistant, tool)*N] conversation."""
    messages = [
        {"role": "system", "content": "system prompt " * 20},
        {"role": "user", "content": "# Task: implement kernelbench_l1_op"},
    ]
    for i in range(num_rounds):
        messages.append(
            {"role": "assistant", "content": f"thinking round {i} " * 10}
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "name": "run_verify",
                "content": "verify output " * (round_chars // 14),
            }
        )
    return messages


class StubChatModel:
    """Mimics OpenAICompatibleChatModel.query for the compactor."""

    def __init__(self, summary_text: str = "## Attempts History\n- 尝试1: ok"):
        self.query = AsyncMock(return_value=(summary_text, [], {"metrics": {}}, {}))
        self.calls: list[list[dict]] = []

    async def _query(self, messages, rollout_cache, **kwargs):
        self.calls.append(messages)
        return self.summary_text, [], rollout_cache, {}

    def __call__(self):
        return self


def stub_model(summary_text: str):
    model = StubChatModel(summary_text)
    model.query = model._query  # plain async method, not AsyncMock wrapper
    model.summary_text = summary_text
    return model


def make_compactor(**overrides) -> ConversationCompactor:
    cfg_kwargs = dict(
        enabled=True,
        threshold_ratio=0.8,
        max_context_tokens=40960,
        min_messages=2,
        failure_backoff_steps=3,
    )
    cfg_kwargs.update(overrides)
    return ConversationCompactor(config=CompactionConfig(**cfg_kwargs))


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


class TestTrigger(unittest.TestCase):
    def test_below_threshold_no_trigger(self):
        compactor = make_compactor()
        messages = make_messages()
        self.assertFalse(compactor.should_compact(messages))

    def test_usage_above_threshold_triggers(self):
        compactor = make_compactor()
        messages = make_messages()
        self.assertTrue(compactor.should_compact(messages, last_usage=(40000, 100)))

    def test_estimate_above_threshold_triggers(self):
        compactor = make_compactor(max_context_tokens=200)  # threshold 160
        messages = make_messages(num_rounds=4, round_chars=2000)
        self.assertTrue(compactor.should_compact(messages))

    def test_disabled_never_triggers(self):
        compactor = make_compactor(enabled=False)
        messages = make_messages()
        self.assertFalse(compactor.should_compact(messages, last_usage=(99000, 0)))

    def test_min_messages_guard(self):
        compactor = make_compactor(min_messages=6)
        messages = make_messages(num_rounds=2)  # 2 + 4 = 6 messages < 2+6
        self.assertFalse(compactor.should_compact(messages, last_usage=(99000, 0)))

    def test_backoff_after_failure(self):
        compactor = make_compactor(failure_backoff_steps=3)
        model = stub_model("")  # empty summary -> rejection path
        messages = make_messages()

        async def run():
            return await compactor.compact(messages, model, step=1)

        self.assertIsNone(asyncio.run(run()))
        # immediately after: backoff active
        self.assertFalse(compactor.should_compact(messages, last_usage=(99000, 0)))
        compactor.notify_step()  # 2
        self.assertFalse(compactor.should_compact(messages, last_usage=(99000, 0)))
        compactor.notify_step()  # 3
        self.assertFalse(compactor.should_compact(messages, last_usage=(99000, 0)))
        compactor.notify_step()  # 4 > backoff_steps
        self.assertTrue(compactor.should_compact(messages, last_usage=(99000, 0)))


# ---------------------------------------------------------------------------
# Compaction: LLM summary path
# ---------------------------------------------------------------------------


class TestCompactSummary(unittest.TestCase):
    def test_rebuild_preserves_system_and_task(self):
        summary = "## Attempts History\n- 尝试1: 直接实现 -> 通过"
        model = stub_model(summary)
        compactor = make_compactor()
        messages = make_messages(num_rounds=4, round_chars=2000)

        new_messages = asyncio.run(compactor.compact(messages, model, step=5))
        self.assertIsNotNone(new_messages)
        self.assertEqual(len(new_messages), 3)
        # system + original user task kept VERBATIM (same object content)
        self.assertEqual(new_messages[0]["role"], "system")
        self.assertEqual(new_messages[0]["content"], messages[0]["content"])
        self.assertEqual(new_messages[1]["role"], "user")
        self.assertEqual(new_messages[1]["content"], messages[1]["content"])
        # summary stored as assistant message with checkpoint preamble
        self.assertEqual(new_messages[2]["role"], "assistant")
        self.assertTrue(new_messages[2]["content"].startswith(CHECKPOINT_PREAMBLE))
        self.assertIn("尝试1", new_messages[2]["content"])
        # original list untouched (caller decides)
        self.assertEqual(len(messages), 10)

    def test_instruction_appended_after_full_history(self):
        model = stub_model("summary text")
        compactor = make_compactor()
        messages = make_messages(num_rounds=3)

        asyncio.run(compactor.compact(messages, model, step=2))

        self.assertEqual(len(model.calls), 1)
        sent = model.calls[0]
        # KV-cache reuse: the full conversation is the prefix...
        self.assertEqual(sent[:-1], messages)
        # ...and the instruction is the LAST user message
        self.assertEqual(sent[-1]["role"], "user")
        self.assertEqual(sent[-1]["content"], COMPACTION_INSTRUCTION)

    def test_summary_smaller_than_shadowed(self):
        model = stub_model("## Attempts History\n- short summary")
        compactor = make_compactor()
        messages = make_messages(num_rounds=4, round_chars=2000)

        new_messages = asyncio.run(compactor.compact(messages, model))
        self.assertIsNotNone(new_messages)
        meter = TokenMeter()
        self.assertLess(
            meter.estimate_messages(new_messages),
            meter.estimate_messages(messages),
        )

    def test_oversized_summary_rejected_after_retries(self):
        # summary longer than everything it shadows -> reject, keep original
        huge_summary = "x" * 20000
        model = stub_model(huge_summary)
        compactor = make_compactor(max_retries=1)
        messages = make_messages(num_rounds=2, round_chars=500)

        result = asyncio.run(compactor.compact(messages, model))
        self.assertIsNone(result)
        # retried once (2 calls), then gave up
        self.assertEqual(len(model.calls), 2)
        stats = compactor.stats()
        self.assertEqual(stats["num_failures"], 1)
        self.assertEqual(stats["events"][-1]["stage"], "rejected")

    def test_empty_summary_rejected(self):
        model = stub_model("   ")
        compactor = make_compactor()
        messages = make_messages()

        result = asyncio.run(compactor.compact(messages, model))
        self.assertIsNone(result)
        self.assertEqual(compactor.stats()["events"][-1]["error"], "empty summary")

    def test_model_error_falls_back(self):
        model = stub_model("ok")
        model.summary_text = "ok"

        async def boom(messages, rollout_cache, **kwargs):
            raise RuntimeError("server down")

        model.query = boom
        compactor = make_compactor()
        messages = make_messages()

        result = asyncio.run(compactor.compact(messages, model))
        self.assertIsNone(result)
        self.assertEqual(compactor.stats()["events"][-1]["stage"], "error")

    def test_frame_summary_strips_code_fences(self):
        framed = frame_summary("```\n## Attempts History\n- x\n```")
        self.assertTrue(framed.startswith(CHECKPOINT_PREAMBLE))
        self.assertNotIn("```", framed)

    def test_extract_compact_result_extracts_tagged_content(self):
        raw = (
            "Let me analyze the conversation...\n\n"
            "<compact_result>\n"
            "## Attempts History\n"
            "- Attempt 1: initial implementation -> compile error\n"
            "## Current State\n"
            "- ast_passed=true\n"
            "</compact_result>\n\n"
            "Task complete."
        )
        extracted = _extract_compact_result(raw)
        self.assertTrue(extracted.startswith("## Attempts History"))
        self.assertNotIn("Let me analyze", extracted)
        self.assertNotIn("Task complete", extracted)
        self.assertIn("- Attempt 1:", extracted)

    def test_extract_compact_result_no_tags_returns_original(self):
        text = "## Attempts History\n- x\n## Current State\n- y"
        self.assertEqual(_extract_compact_result(text), text)

    def test_extract_compact_result_empty_tags(self):
        self.assertEqual(_extract_compact_result("<compact_result></compact_result>"), "")

    def test_frame_summary_extracts_tags_before_wrapping(self):
        raw = (
            "Let me analyze...\n\n"
            "<compact_result>\n"
            "## Attempts History\n- Attempt 1: ok\n"
            "## Current State\n- done\n"
            "</compact_result>"
        )
        framed = frame_summary(raw)
        self.assertTrue(framed.startswith(CHECKPOINT_PREAMBLE))
        self.assertNotIn("Let me analyze", framed)
        self.assertNotIn("<compact_result>", framed)
        self.assertIn("## Attempts History", framed)
        self.assertIn("## Current State", framed)

    def test_last_summary_call_records_prompt_and_response(self):
        model = stub_model("## Attempts History\n- summary body")
        compactor = make_compactor()
        messages = make_messages(num_rounds=3)

        new_messages = asyncio.run(compactor.compact(messages, model))
        self.assertIsNotNone(new_messages)

        call = compactor.last_summary_call
        self.assertIsNotNone(call)
        # prompt = exact summarizer input: conversation + instruction last
        self.assertEqual(call["prompt_messages"][-1]["role"], "user")
        self.assertEqual(call["prompt_messages"][-1]["content"], COMPACTION_INSTRUCTION)
        self.assertEqual(call["prompt_messages"][0], messages[0])
        # response = the framed checkpoint text (thinking stripped)
        self.assertEqual(call["summary"], new_messages[2]["content"])
        self.assertTrue(call["summary"].startswith(CHECKPOINT_PREAMBLE))
        # raw_response = model's original output (before thinking strip)
        self.assertEqual(call["raw_response"], "## Attempts History\n- summary body")

    def test_multi_compaction_single_checkpoint(self):
        model = stub_model("## Attempts History\n- merged checkpoint")
        compactor = make_compactor()
        messages = make_messages(num_rounds=4, round_chars=2000)

        first = asyncio.run(compactor.compact(messages, model, step=3))
        self.assertIsNotNone(first)

        # simulate continued work after first compaction
        continued = first + [
            {"role": "assistant", "content": "more work"},
            {"role": "tool", "tool_call_id": "c", "name": "run_verify",
             "content": "still working"},
        ]
        second = asyncio.run(compactor.compact(continued, model, step=9))
        self.assertIsNotNone(second)
        # still exactly ONE checkpoint message, at most 3 messages rebuilt
        self.assertEqual(len(second), 3)
        checkpoints = [
            m for m in second if m["role"] == "assistant"
        ]
        self.assertEqual(len(checkpoints), 1)
        stats = compactor.stats()
        self.assertEqual(stats["num_compactions"], 2)


# ---------------------------------------------------------------------------
# Segment workspace snapshot
# ---------------------------------------------------------------------------


class StubFileEnv:
    """Env stand-in whose read_file serves a dict of path -> text."""

    def __init__(self, files: dict[str, str] | None = None, raise_on_read: bool = False):
        self.files = files or {}
        self.raise_on_read = raise_on_read

    async def read_file(self, path):
        if self.raise_on_read:
            raise RuntimeError("sandbox unreachable")
        if path in self.files:
            return self.files[path]
        raise FileNotFoundError(path)


class TestResumeNote(unittest.TestCase):
    def test_extracts_next_step_section(self):
        from examples.triton_agent.synth_common import (
            COMPACTION_RESUME_NOTE,
            _extract_next_step_section,
        )

        checkpoint = (
            "## Current State\n- ast_passed=true\n\n"
            "## Next Step\n- run run_verify on latest config\n- then submit\n\n"
            "## Design Decisions\n- blocked tiling"
        )
        self.assertEqual(
            _extract_next_step_section(checkpoint),
            "- run run_verify on latest config\n- then submit",
        )
        # no Next Step section -> empty (caller falls back to a pointer)
        self.assertEqual(_extract_next_step_section("## Current State\n- x"), "")
        # template is formattable
        self.assertIn("run_verify", COMPACTION_RESUME_NOTE.format(next_step="- run run_verify"))


class TestSegmentWorkspaceSnapshot(unittest.TestCase):
    WS = "/opt/workspace_test/agent_workdir"
    OP = "kernelbench_l1_test_op"

    def _impl_path(self):
        return f"{self.WS}/src/{self.OP}_triton_ascend_impl.py"

    def _verify_path(self):
        return f"{self.WS}/output/verify/verify_result.json"

    def _perf_path(self):
        return f"{self.WS}/output/perf_result.json"

    def _perf_best_path(self):
        return f"{self.WS}/output/perf_result_best.json"

    def test_snapshot_collects_impl_and_verify(self):
        from examples.triton_agent.synth_common import _segment_workspace_snapshot

        env = StubFileEnv(
            {
                self._impl_path(): "class ModelNew: ...",
                self._verify_path(): '{"pass_rate": 1.0, "cases": []}',
            }
        )
        snapshot = asyncio.run(_segment_workspace_snapshot(env, self.WS, self.OP))
        self.assertEqual(snapshot["impl_file"], "class ModelNew: ...")
        self.assertIsInstance(snapshot["verify_result"], dict)
        self.assertEqual(snapshot["verify_result"]["pass_rate"], 1.0)

    def test_snapshot_collects_perf_current_and_best(self):
        from examples.triton_agent.synth_common import _segment_workspace_snapshot

        env = StubFileEnv(
            {
                self._impl_path(): "class ModelNew: ...",
                self._verify_path(): '{"pass_rate": 1.0, "cases": []}',
                self._perf_path(): '{"speedup_vs_torch": 1.5, "total_cases": 4, "passed_cases": 4, "implementation": {"avg_latency_ms": 10.0}, "framework": {"avg_latency_ms": 15.0}}',
                self._perf_best_path(): '{"speedup_vs_torch": 2.0, "total_cases": 4, "passed_cases": 4, "implementation": {"avg_latency_ms": 8.0}, "framework": {"avg_latency_ms": 16.0}}',
            }
        )
        snapshot = asyncio.run(_segment_workspace_snapshot(env, self.WS, self.OP))
        self.assertEqual(snapshot["perf_current"]["speedup_vs_torch"], 1.5)
        self.assertEqual(snapshot["perf_current"]["total_cases"], 4)
        self.assertEqual(snapshot["perf_current"]["implementation"]["avg_latency_ms"], 10.0)
        self.assertEqual(snapshot["perf_best"]["speedup_vs_torch"], 2.0)
        self.assertEqual(snapshot["perf_best"]["implementation"]["avg_latency_ms"], 8.0)

    def test_snapshot_partial_when_verify_missing(self):
        from examples.triton_agent.synth_common import _segment_workspace_snapshot

        env = StubFileEnv({self._impl_path(): "impl code"})
        snapshot = asyncio.run(_segment_workspace_snapshot(env, self.WS, self.OP))
        self.assertEqual(snapshot, {"impl_file": "impl code"})

    def test_snapshot_empty_and_never_raises(self):
        from examples.triton_agent.synth_common import _segment_workspace_snapshot

        # nothing written yet -> empty dict
        snapshot = asyncio.run(_segment_workspace_snapshot(StubFileEnv(), self.WS, self.OP))
        self.assertEqual(snapshot, {})
        # sandbox errors are swallowed too
        snapshot = asyncio.run(
            _segment_workspace_snapshot(StubFileEnv(raise_on_read=True), self.WS, self.OP)
        )
        self.assertEqual(snapshot, {})


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class TestObservability(unittest.TestCase):
    def test_stats_shape(self):
        model = stub_model("## Attempts History\n- x")
        compactor = make_compactor()
        messages = make_messages()
        result = asyncio.run(compactor.compact(messages, model, step=7))
        self.assertIsNotNone(result)

        stats = compactor.stats()
        self.assertEqual(stats["num_compactions"], 1)
        event = stats["events"][0]
        self.assertEqual(event["step"], 7)
        self.assertEqual(event["stage"], "summary")
        self.assertEqual(event["messages_before"], len(messages))
        self.assertEqual(event["messages_after"], 3)
        self.assertGreater(event["tokens_before"], event["tokens_after"])
        self.assertGreater(event["summary_chars"], 0)
        self.assertGreaterEqual(event["duration_s"], 0.0)


if __name__ == "__main__":
    unittest.main()