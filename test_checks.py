"""
Tests for the parts that must never break: parsing and the timestamp checks that stop
the model from inventing or stretching clips. Run: python test_checks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framer import apply_cut, frame                          # noqa: E402
from llm_nodes import LLM, Hook, HookList, fmt               # noqa: E402
from parser import parse_episode, to_seconds                 # noqa: E402

CFG = {"min_seconds": 20, "max_seconds": 60, "hook_must_land_within": 3.0,
       "snap_tolerance": 1.0}

SCRIPT = "\n".join(
    ["EPISODE NUMBER: Episode 7", "EPISODE LOGLINE: test", "CONTINUITY NOTES: test", "",
     "SCENE 01", "Scene heading: INT. HALL – DAY", "Scene start: 00:00.00",
     "Scene end: 01:30.00", ""]
    + sum([[r"\*\*CAMERA:\*\*", f"Start: ({fmt(t)}) | End: ({fmt(t + 5)})", "Shot.", "",
            f"**HERO (Angry):**", f"Start: ({fmt(t + 1)}) | End: ({fmt(t + 4)})",
            f'"Line at {t}\\!"', ""] for t in range(0, 90, 5)], []))
EP = parse_episode(SCRIPT, "episode-007-x.txt")


def test_parser():
    assert EP["episode"] == 7 and EP["duration"] == 90.0
    assert to_seconds("01:05.60") == 65.6
    d = [e for e in EP["events"] if e["type"] == "dialogue"][0]
    assert d["speaker"] == "HERO" and d["emotion"] == "Angry" and d["text"] == "Line at 0!"


def test_invented_hook_timestamp_rejected():
    clip, why = frame(EP, Hook(start_timestamp="00:12.50", hook_type="x", hook_description="x"), 1, CFG)
    assert clip is None and "matches no line" in why


def test_hook_snaps_and_lands_in_first_3s():
    clip, _ = frame(EP, Hook(start_timestamp="00:11.20", hook_type="x", hook_description="x"), 1, CFG)
    assert clip["hook_at"] == 11.0 and clip["start"] == 10.0     # starts on the camera cut
    assert clip["hook_at"] - clip["start"] <= 3


def test_hook_too_late_rejected():
    clip, why = frame(EP, Hook(start_timestamp="01:16.00", hook_type="x", hook_description="x"), 1, CFG)
    assert clip is None and "only" in why


def test_end_out_of_range_is_clamped():
    clip, _ = frame(EP, Hook(start_timestamp="00:06.00", hook_type="x", hook_description="x"), 1, CFG)
    end, note = apply_cut(clip, "01:40.00", EP, CFG)              # model asked for 95s
    assert clip["start"] + 20 <= end <= clip["start"] + 60 and "outside" in note
    end, note = apply_cut(clip, "00:09.00", EP, CFG)              # model asked for 4s
    assert end - clip["start"] >= 20 and "outside" in note


def test_end_snaps_to_real_line_end():
    clip, _ = frame(EP, Hook(start_timestamp="00:06.00", hook_type="x", hook_description="x"), 1, CFG)
    end, note = apply_cut(clip, "00:39.20", EP, CFG)
    assert end == 39.0 and note == ""


def test_garbage_end_rejected():
    clip, _ = frame(EP, Hook(start_timestamp="00:06.00", hook_type="x", hook_description="x"), 1, CFG)
    end, note = apply_cut(clip, "around the middle", EP, CFG)
    assert end is None and "not a valid time" in note


def test_llm_retries_bad_json_then_caches(tmp_path=None):
    import tempfile
    d = tmp_path or Path(tempfile.mkdtemp())
    llm = LLM({"provider": "gemini", "model": "fake", "requests_per_minute": 6000,
               "max_retries": 3}, str(d))
    replies = iter(["not json", '```json\n{"hooks": []}\n```'])
    llm._call = lambda p, s, v=None: next(replies)
    import time
    real_sleep, time.sleep = time.sleep, lambda s: None           # skip backoff waits
    try:
        assert llm.structured("p", HookList, "t").hooks == []
        assert llm.stats["retries"] == 1
        llm._call = lambda p, s, v=None: (_ for _ in ()).throw(AssertionError("should hit cache"))
        assert llm.structured("p", HookList, "t").hooks == []
        assert llm.stats["cache_hits"] == 1
    finally:
        time.sleep = real_sleep


def test_llm_gives_up_cleanly_on_rate_limit():
    import tempfile, time
    llm = LLM({"provider": "gemini", "model": "fake2", "requests_per_minute": 6000,
               "max_retries": 2}, tempfile.mkdtemp())
    llm._call = lambda p, s, v=None: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED"))
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        assert llm.structured("p", HookList, "t") is None and llm.stats["failed_calls"] == 1
    finally:
        time.sleep = real_sleep


def _fake(llm, script):
    """Make llm._call follow a script: an Exception is raised, a str is returned."""
    it = iter(script)
    def call(p, s, v=None):
        llm.stats["calls_by_model"][llm.model] = llm.stats["calls_by_model"].get(llm.model, 0) + 1
        x = next(it)
        if isinstance(x, Exception):
            raise x
        return x
    llm._call = call


def test_overloaded_model_switches_to_fallback():
    import tempfile, time
    llm = LLM({"provider": "gemini", "model": "busy-model", "fallback_models": ["backup"],
               "requests_per_minute": 6000, "max_retries": 6}, tempfile.mkdtemp())
    busy = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
    _fake(llm, [busy, busy, '{"hooks": []}'])
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        assert llm.structured("p", HookList, "t").hooks == []
        assert llm.model == "backup" and llm.stats["calls_by_model"] == {"busy-model": 2, "backup": 1}
    finally:
        time.sleep = real_sleep


def test_quota_switches_at_once_then_stops_cleanly():
    import tempfile, time
    from llm_nodes import QuotaExhausted
    llm = LLM({"provider": "gemini", "model": "a", "fallback_models": ["b"],
               "requests_per_minute": 6000, "max_retries": 2}, tempfile.mkdtemp())
    quota = RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota")
    _fake(llm, [quota] * 20)
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        assert llm.structured("p1", HookList, "t1") is None          # a -> b, b fails
        assert llm.model == "b"
        try:
            llm.structured("p2", HookList, "t2")                     # 2nd failed call -> stop
            assert False, "should have raised"
        except QuotaExhausted:
            pass
    finally:
        time.sleep = real_sleep


def test_cache_from_fallback_model_is_reused():
    import tempfile, time
    d = tempfile.mkdtemp()
    llm = LLM({"provider": "gemini", "model": "a", "fallback_models": ["b"],
               "requests_per_minute": 6000, "max_retries": 3}, d)
    _fake(llm, [RuntimeError("429 quota"), '{"hooks": []}'])
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        llm.structured("p", HookList, "t")
        again = LLM({"provider": "gemini", "model": "a", "fallback_models": ["b"],
                     "requests_per_minute": 6000}, d)
        _fake(again, [])                                             # any call would fail
        assert again.structured("p", HookList, "t").hooks == [] and again.stats["cache_hits"] == 1
    finally:
        time.sleep = real_sleep



def test_claude_backend_parses_tool_call():
    """Claude path with a fake SDK: forced tool call -> schema object, usage counted."""
    import tempfile, types as T
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            block = T.SimpleNamespace(type="tool_use", input={"hooks": [
                {"start_timestamp": "00:18.00", "hook_type": "public order",
                 "hook_description": "Rupert screams for the guards"}]})
            return T.SimpleNamespace(content=[block], stop_reason="tool_use",
                                     usage=T.SimpleNamespace(input_tokens=1200, output_tokens=80))

    fake = T.ModuleType("anthropic")
    fake.Anthropic = lambda **kw: T.SimpleNamespace(messages=FakeMessages())
    sys.modules["anthropic"] = fake
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test")
    llm = LLM({"provider": "anthropic", "model": "claude-sonnet-5-5",
               "requests_per_minute": 6000}, tempfile.mkdtemp())
    out = llm.structured("find hooks", HookList, "t")
    assert out.hooks[0].start_timestamp == "00:18.00"
    assert seen["tool_choice"] == {"type": "tool", "name": "submit_answer"}
    assert "$ref" not in str(seen["tools"][0]["input_schema"])
    assert llm.stats["input_tokens"] == 1200 and llm.stats["calls_by_model"] == {"claude-sonnet-5-5": 1}


def test_no_credits_stops_run_immediately():
    import tempfile, time
    from llm_nodes import QuotaExhausted
    llm = LLM({"provider": "anthropic", "model": "m", "requests_per_minute": 6000}, tempfile.mkdtemp())
    _fake(llm, [RuntimeError("Error code: 400 - Your credit balance is too low to access the API")])
    try:
        llm.structured("p", HookList, "t")
        assert False, "should stop"
    except QuotaExhausted as e:
        assert "credits" in str(e)



def test_openai_backend_strict_schema_and_temperature_retry():
    """ChatGPT path with a fake SDK: strict schema sent; 'temperature unsupported' -> retried without it."""
    import tempfile, types as T, os
    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw)
            if "temperature" in kw:
                raise RuntimeError("Error code: 400 - Unsupported value: 'temperature' does not "
                                   "support 0.2 with this model. Only the default (1) is supported.")
            msg = T.SimpleNamespace(content='{"hooks": [{"start_timestamp": "00:11.30", '
                                    '"hook_type": "humiliation", "hook_description": "x"}]}', refusal=None)
            return T.SimpleNamespace(choices=[T.SimpleNamespace(message=msg)],
                                     usage=T.SimpleNamespace(prompt_tokens=900, completion_tokens=60))

    fake = T.ModuleType("openai")
    fake.OpenAI = lambda **kw: T.SimpleNamespace(chat=T.SimpleNamespace(completions=FakeCompletions()))
    sys.modules["openai"] = fake
    os.environ.setdefault("OPENAI_API_KEY", "test")
    llm = LLM({"provider": "openai", "model": "gpt-5-mini", "requests_per_minute": 6000},
              tempfile.mkdtemp())
    out = llm.structured("find hooks", HookList, "t")
    assert out.hooks[0].start_timestamp == "00:11.30"
    assert len(calls) == 2 and "temperature" not in calls[1]
    sch = calls[1]["response_format"]["json_schema"]
    assert sch["strict"] is True and "$ref" not in str(sch["schema"])
    item = sch["schema"]["properties"]["hooks"]["items"]
    assert item["additionalProperties"] is False and set(item["required"]) == {
        "start_timestamp", "hook_type", "hook_description"}
    llm.structured("again", HookList, "t2")                 # remembers: no temperature next time
    assert "temperature" not in calls[-1] and len(calls) == 3


def test_openai_out_of_credit_stops_but_gemini_rate_limit_does_not():
    from llm_nodes import _error_kind
    assert _error_kind("Error code: 429 - insufficient_quota: You exceeded your current quota") == "billing"
    assert _error_kind("429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check "
                       "your plan and billing details") == "quota"



def test_manual_mode_round_trip():
    """Manual mode: task -> batch prompt -> pasted chat reply (with fences) -> answer reused."""
    import tempfile, json as J
    import manual
    d = tempfile.mkdtemp()
    llm = LLM({"provider": "manual", "model": "ChatGPT (web)", "manual_dir": d}, d)
    assert llm.structured("find hooks in ep 1", HookList, "hooks-ep001") is None   # pending
    assert llm.structured("find hooks in ep 2", HookList, "hooks-ep002") is None
    bs = manual.write_batches(llm.pending, d, 40000)
    assert len(bs) == 1 and bs[0]["tags"] == ["hooks-ep001", "hooks-ep002"]
    text = manual.batches(d)[0]["text"]
    assert "=== TASK hooks-ep001 ===" in text and '"start_timestamp"' in text
    hook = {"start_timestamp": "00:11.30", "hook_type": "humiliation", "hook_description": "x"}
    saved, problems = manual.save_answer(d, "batch_01", "not json at all")
    assert saved == [] and "JSON" in problems[0]
    reply = "Here you go:\n```json\n" + J.dumps({"hooks-ep001": {"hooks": [hook]}}) + "\n```"
    saved, problems = manual.save_answer(d, "batch_01", reply)
    assert saved == ["hooks-ep001"] and "hooks-ep002: missing" in problems[0]
    again = LLM({"provider": "manual", "model": "x", "manual_dir": d}, d)
    assert again.structured("find hooks in ep 1", HookList, "hooks-ep001").hooks[0].start_timestamp == "00:11.30"
    assert again.structured("CHANGED prompt for ep 1", HookList, "hooks-ep001") is None   # stale answer not reused


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
