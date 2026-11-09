"""
LLM nodes - Phase 2 (Hook Hunter), Phase 4 (Judge & Trimmer), Phase 5 (Ranker).

All model calls go through one `LLM` class, so the model is set in config.yaml:
  provider: openai             -> ChatGPT / OpenAI API (openai SDK, strict JSON schema)
  provider: anthropic          -> Claude (anthropic SDK)
  provider: gemini             -> Google Gemini (google-genai SDK)
  provider: openai_compatible  -> Groq, OpenRouter, OpenAI, Ollama, ... (openai SDK)
  provider: manual             -> you paste prompts into a chat model and paste replies back
  provider: rules              -> no LLM: script signals only, no API key ('mock' = same)

Every response is cached on disk (keyed by model + prompt), so a rerun of the same
show costs zero calls. Calls are paced to `requests_per_minute` and retried with
exponential backoff on rate-limit / server errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, ValidationError

from parser import to_seconds


# =============================================================================
# Schemas (what the model must return)
# =============================================================================

class Hook(BaseModel):
    start_timestamp: str          # copied from the script, "MM:SS.ss"
    hook_type: str                # e.g. slap, public humiliation, identity reveal
    hook_description: str         # the moment in a few words


class HookList(BaseModel):
    hooks: list[Hook]


class PromoJudgeEvaluation(BaseModel):
    """The locked V3 schema, plus the 4 text fields marked [brief] that Kuku's brief
    requires for every promo (the hook, the cliffhanger, one line on why it works)."""
    is_valid_promo: bool
    # THE DYNAMIC CUT
    recommended_end_timestamp: str        # must be between +20s and +60s (Python checks)
    # NARRATIVE GRADING (1-10)
    hook_intensity_score: int
    cliffhanger_intensity_score: int
    middle_story_engagement_score: int
    # CONTEXT & QUALITY CONTROL
    is_self_understandable: bool          # the Amnesia Test
    missing_context_explanation: str
    # [brief] required outputs
    title: str
    hook_text: str
    cliffhanger_text: str
    why_it_works: str
    # V1 archetypes, filled from rules.json when the Rule Maker has been run
    hook_archetype: str
    cliffhanger_archetype: str


class RankedClip(BaseModel):
    clip_id: str
    reason: str


class Ranking(BaseModel):
    ranking: list[RankedClip]


# =============================================================================
# Model wrapper
# =============================================================================

class QuotaExhausted(RuntimeError):
    """The run cannot continue (out of quota/credits, bad key). Finished calls are cached."""


def _error_kind(err: str) -> str:
    e = err.lower()
    if "credit balance" in e or "insufficient_quota" in e:
        return "billing"
    if "401" in e or "invalid x-api-key" in e or "api key not valid" in e or "authentication" in e:
        return "auth"
    if "429" in e or "resource_exhausted" in e or "quota" in e or "rate limit" in e:
        return "quota"
    if any(s in e for s in ("503", "529", "unavailable", "overloaded", "high demand", "500 ",
                            "502", "504", "internal")):
        return "busy"
    if "timeout" in e or "timed out" in e or "disconnected" in e or "remoteprotocol" in e:
        return "network"
    return "other"


class LLM:
    def __init__(self, cfg: dict, cache_dir: str = ".cache"):
        self.cfg = cfg
        self.provider = cfg["provider"]
        known = ("openai", "anthropic", "gemini", "openai_compatible", "manual", "rules", "mock")
        if self.provider not in known:      # a config typo should fail at once, not retry
            raise SystemExit(f"Unknown provider '{self.provider}'. Use one of: {', '.join(known)}.")
        self.pending: list[dict] = []           # manual mode: tasks waiting for a pasted reply
        self.manual_dir = Path(cfg.get("manual_dir") or "manual")
        if self.provider in ("rules", "mock"):     # no LLM: script signals only
            self.models = ["rules (no LLM)"]
        elif self.provider == "manual":            # a person pastes prompts into a chat model
            self.models = [f"manual: {cfg.get('model') or 'chat model'}"]
        else:   # main model first, then backups used when it is overloaded / out of quota
            fb = cfg.get("fallback_models") or []
            self.models = [cfg["model"]] + [m for m in fb if m and m != cfg["model"]]
        self.mi = 0
        self.min_interval = 60.0 / max(1, cfg.get("requests_per_minute", 8))
        self.max_retries = cfg.get("max_retries", 6)
        self.timeout = cfg.get("timeout_seconds", 180)
        self.cache_root = Path(cache_dir) / self.provider
        self._last = 0.0
        self._client = None
        self._quota_strikes = 0
        self._no_temperature = False     # GPT-5 family only accepts the default temperature
        self.stats = {"api_calls": 0, "cache_hits": 0, "failed_calls": 0, "retries": 0,
                      "input_tokens": 0, "output_tokens": 0, "seconds_waiting": 0.0,
                      "calls_by_model": {}, "model_switches": []}

    @property
    def model(self) -> str:
        return self.models[self.mi]

    def _cache_file(self, model: str, tag: str, body: str) -> Path:
        d = self.cache_root / re.sub(r"[^\w.-]", "_", model)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{tag}-{hashlib.sha256(f'{model}|{body}'.encode()).hexdigest()[:24]}.json"

    # ---- public --------------------------------------------------------------
    def structured(self, prompt: str, schema: type[BaseModel], tag: str,
                   mock: Optional[Callable[[], BaseModel]] = None,
                   video_url: Optional[str] = None) -> Optional[BaseModel]:
        """Return an instance of `schema`, or None if every retry failed.
        `video_url`: a public YouTube link the model should watch (Gemini only).
        Raises QuotaExhausted when all models keep returning 'out of quota'."""
        if self.provider in ("rules", "mock"):
            return mock() if mock else None
        if self.provider == "manual":
            if video_url:
                raise ValueError("manual mode cannot watch videos")
            key = f"{tag}-{hashlib.sha256(prompt.encode()).hexdigest()[:10]}"   # answer <-> exact prompt
            ans = self.manual_dir / "answers" / f"{key}.json"
            if ans.exists():
                try:
                    obj = schema.model_validate_json(ans.read_text())
                    self.stats["cache_hits"] += 1
                    return obj
                except ValidationError:
                    ans.unlink()
            self.pending.append({"tag": tag, "key": key, "schema": schema.__name__, "prompt": prompt})
            return None
        if video_url and self.provider != "gemini":
            raise ValueError("watching YouTube links needs provider: gemini")

        body = f"{schema.__name__}|{video_url}|{prompt}"
        for m in self.models:                          # reuse an answer from any listed model
            path = self._cache_file(m, tag, body)
            if path.exists():
                try:
                    obj = schema.model_validate_json(path.read_text())
                    self.stats["cache_hits"] += 1
                    return obj
                except ValidationError:
                    path.unlink()                      # stale cache -> call again

        kind = "other"
        for attempt in range(self.max_retries):
            self._pace()
            try:
                text = self._call(prompt, schema, video_url)
                obj = schema.model_validate_json(_strip_fences(text))
                self._cache_file(self.model, tag, body).write_text(obj.model_dump_json(indent=2))
                self._quota_strikes = 0
                return obj
            except ValidationError as e:               # model returned bad JSON
                last, kind = f"invalid JSON: {str(e)[:160]}", "json"
            except Exception as e:                     # quota, overload, network
                last = f"{type(e).__name__}: {str(e)[:160]}"
                kind = _error_kind(last)
            self.stats["retries"] += 1
            if kind == "billing":
                raise QuotaExhausted(f"{self.model}: account has no credits left - add credits "
                                     f"or switch provider. ({last})")
            if kind == "auth":
                raise QuotaExhausted(f"API key rejected - check the {self._key_env()} secret. ({last})")

            # Out of quota -> switch at once; overloaded -> switch after 2 tries.
            if (kind == "quota" or (kind == "busy" and attempt >= 1)) \
                    and self.mi < len(self.models) - 1:
                old = self.model
                self.mi += 1
                self.stats["model_switches"].append(f"{old} -> {self.model} ({kind})")
                print(f"   [{tag}] {old} is {'out of quota' if kind == 'quota' else 'overloaded'}; "
                      f"switching to {self.model}")
                continue

            if kind == "quota":
                wait = 65                              # per-minute quotas reset after a minute
            elif kind == "busy":
                wait = min(90, 10 * 2 ** attempt)
            else:
                wait = min(60, 3 * 2 ** attempt)
            wait += random.random()
            print(f"   [{tag}] attempt {attempt + 1} failed ({last}); waiting {wait:.0f}s")
            self.stats["seconds_waiting"] += wait
            time.sleep(wait)

        self.stats["failed_calls"] += 1
        if kind == "quota":
            self._quota_strikes += 1
            if self._quota_strikes >= 2:
                raise QuotaExhausted(
                    f"All models ({', '.join(self.models)}) are out of quota. Finished calls are "
                    "saved in .cache - rerun later, or add another model with --fallback.")
        return None

    # ---- internals -----------------------------------------------------------
    def _pace(self):
        gap = time.time() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
            self.stats["seconds_waiting"] += self.min_interval - gap
        self._last = time.time()

    def _key_env(self) -> str:
        return self.cfg.get("api_key_env") or {"anthropic": "ANTHROPIC_API_KEY",
                                               "gemini": "GEMINI_API_KEY"}.get(self.provider,
                                                                                "OPENAI_API_KEY")

    def _key(self) -> str:
        env = self._key_env()
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"Set the {env} environment variable (your API key).")
        return key

    def _call(self, prompt: str, schema: type[BaseModel], video_url: Optional[str] = None) -> str:
        self.stats["api_calls"] += 1
        self.stats["calls_by_model"][self.model] = self.stats["calls_by_model"].get(self.model, 0) + 1
        if self.provider == "anthropic":
            import anthropic
            if self._client is None:   # our own retry loop handles retries
                self._client = anthropic.Anthropic(api_key=self._key(), timeout=self.timeout,
                                                   max_retries=0)
            tool = {"name": "submit_answer", "description": "Submit the answer in this exact shape.",
                    "input_schema": _inline_refs(schema.model_json_schema())}
            r = self._client.messages.create(
                model=self.model,
                max_tokens=self.cfg.get("max_output_tokens", 4096),
                temperature=self.cfg.get("temperature", 0.2),
                tools=[tool],
                tool_choice={"type": "tool", "name": "submit_answer"},   # forces the schema
                messages=[{"role": "user", "content": prompt}],
            )
            if r.usage:
                self.stats["input_tokens"] += r.usage.input_tokens or 0
                self.stats["output_tokens"] += r.usage.output_tokens or 0
            block = next((b for b in r.content if b.type == "tool_use"), None)
            if block is None:
                raise ValueError(f"no tool_use block (stop_reason={r.stop_reason})")
            return json.dumps(block.input)

        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            if self._client is None:   # timeout so a hung request is retried, not stuck forever
                self._client = genai.Client(
                    api_key=self._key(),
                    http_options=types.HttpOptions(timeout=int(self.timeout * 1000)))
            contents = prompt
            if video_url:   # Gemini fetches and watches public YouTube videos itself
                contents = [types.Part(file_data=types.FileData(file_uri=video_url)),
                            types.Part(text=prompt)]
            r = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.cfg.get("temperature", 0.2),
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            u = getattr(r, "usage_metadata", None)
            if u:
                self.stats["input_tokens"] += u.prompt_token_count or 0
                self.stats["output_tokens"] += u.candidates_token_count or 0
            return r.text

        if self.provider in ("openai", "openai_compatible"):
            from openai import OpenAI
            if self._client is None:
                self._client = OpenAI(base_url=self.cfg.get("base_url"), api_key=self._key(),
                                      timeout=self.timeout, max_retries=0)
            if self.provider == "openai":   # OpenAI structured outputs: answer must match schema
                kw = {"messages": [{"role": "user", "content": prompt}],
                      "response_format": {"type": "json_schema", "json_schema": {
                          "name": schema.__name__, "strict": True,
                          "schema": _strict(_inline_refs(schema.model_json_schema()))}}}
            else:                           # other OpenAI-style APIs: JSON mode + schema in prompt
                kw = {"response_format": {"type": "json_object"}, "messages": [
                    {"role": "system", "content": "Reply with ONE JSON object matching this "
                     f"JSON schema, nothing else:\n{json.dumps(schema.model_json_schema())}"},
                    {"role": "user", "content": prompt}]}
            if not self._no_temperature:
                kw["temperature"] = self.cfg.get("temperature", 0.2)
            try:
                r = self._client.chat.completions.create(model=self.model, **kw)
            except Exception as e:
                if "temperature" not in str(e).lower() or "temperature" not in kw:
                    raise
                self._no_temperature = True         # model rejects it: use its default
                kw.pop("temperature")
                r = self._client.chat.completions.create(model=self.model, **kw)
            if r.usage:
                self.stats["input_tokens"] += r.usage.prompt_tokens or 0
                self.stats["output_tokens"] += r.usage.completion_tokens or 0
            msg = r.choices[0].message
            if getattr(msg, "refusal", None):
                raise ValueError(f"model refused: {msg.refusal}")
            return msg.content

        raise ValueError(f"unknown provider: {self.provider}")


def _inline_refs(schema: dict) -> dict:
    """Replace Pydantic's $defs/$ref with the definitions themselves (plain JSON schema)."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk(schema)


def _strict(schema: dict) -> dict:
    """OpenAI strict mode: every object lists all its fields as required and allows no others."""
    if isinstance(schema, dict):
        out = {k: _strict(v) for k, v in schema.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(schema, list):
        return [_strict(x) for x in schema]
    return schema


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return text


# =============================================================================
# Helpers to show the script to the model
# =============================================================================

def fmt(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


def render(events: list[dict]) -> str:
    """One line per event, with the exact timestamps the model must copy."""
    out = []
    for e in events:
        span = f"[{fmt(e['start'])}-{fmt(e['end'])}]"
        if e["type"] == "dialogue":
            who = e["speaker"] + (" (off-screen)" if e["offscreen"] else "")
            emo = f" ({e['emotion']})" if e["emotion"] else ""
            out.append(f'{span} {who}{emo}: "{e["text"]}"')
        else:
            out.append(f"{span} {e['type'].upper()}: {e['text']}")
    return "\n".join(out)


# =============================================================================
# Rubric from the Rule Maker (rules.json), shared by Hook Hunter and Judge
# =============================================================================

RUBRIC = {"text": "", "names": ""}


def load_rubric(path: str = "rules.json") -> bool:
    """Read rules.json (written by rule_maker.py) into the prompts. Returns False if absent."""
    p = Path(path)
    if not p.exists():
        RUBRIC.update(text="", names="")
        return False
    r = json.loads(p.read_text())
    s = r["stats"]["all"]
    lines = [f"\nWHAT WORKS IN {s['promos_analysed']} REAL PROMOS (ReelShort, DramaBox):",
             f"- Length: median {s['duration_median']}s, middle half {s['duration_p25']}-"
             f"{s['duration_p75']}s; {s['share_one_continuous_stretch']:.0%} are one continuous stretch.",
             "- Hook patterns (name: what it looks like in a script | example):"]
    lines += [f"  * {h['name']}: {h['description']} | e.g. {h['example']}" for h in r["hook_archetypes"]]
    lines += ["- Cliffhanger patterns:"]
    lines += [f"  * {c['name']}: {c['description']} | e.g. {c['example']}"
              for c in r["cliffhanger_archetypes"]]
    lines += [f"- {o}" for o in r.get("other_patterns", [])]
    names = ("the closest pattern name from the lists above (hooks: "
             + ", ".join(h["name"] for h in r["hook_archetypes"]) + "; cliffhangers: "
             + ", ".join(c["name"] for c in r["cliffhanger_archetypes"])
             + "), or 'other: <2-4 words>' if none fits.")
    RUBRIC.update(text="\n".join(lines) + "\n", names=names)
    return True


def _archetype_rule() -> str:
    return RUBRIC["names"] or "a 2-4 word label for each (e.g. 'public slap', 'unanswered question')."


# =============================================================================
# Phase 2 - Hook Hunter (one call per episode)
# =============================================================================

HOOK_PROMPT = """You find promo hooks in a vertical microdrama for a US audience on TikTok,
Instagram Reels and YouTube Shorts.

A HOOK is a moment that makes a scrolling stranger stop within 3 seconds:
a slap or physical attack, public humiliation, a betrayal line, an identity reveal,
a shocking accusation, someone being dragged/seized, a proposal or kiss out of nowhere.
Calm establishing shots, slow walking and small talk are NOT hooks.
{rubric}
Rules:
- Return {count} hooks, best first. Fewer is fine if the episode is weak.
- start_timestamp MUST be copied exactly from the START time of one line below (format MM:SS.ss).
- Pick the line or action that CAUSES the moment (the slap, the line), not the reaction shot after it.
- Each hook needs at least 20 seconds of story after it inside this episode
  (the episode ends at {end}), so do not pick anything after {latest}.
- Hooks should be at least 10 seconds apart.

EPISODE {episode}
Logline: {logline}

SCRIPT (each line: [start-end] who/what: text)
{script}
"""


def hook_hunter(llm: LLM, ep: dict, n_max: int = 5) -> list[Hook]:
    latest = max(0.0, ep["duration"] - 20)
    prompt = HOOK_PROMPT.format(
        count=f"3 to {n_max}" if n_max > 3 else f"up to {n_max}", rubric=RUBRIC["text"],
        episode=ep["episode"], logline=ep["logline"],
        end=fmt(ep["duration"]), latest=fmt(latest), script=render(ep["events"]))
    res = llm.structured(prompt, HookList, f"hooks-ep{ep['episode']:03d}",
                         mock=lambda: _mock_hooks(ep, n_max))
    return res.hooks if res else []


# =============================================================================
# Phase 4 - Promo Judge & Trimmer (one call per candidate clip)
# =============================================================================

JUDGE_PROMPT = """You are a promo editor for a vertical microdrama app. Below is a
{window:.0f}-second stretch of one episode. A promo clip STARTS at {start}
(the hook: {hook}). You decide where it ENDS and grade it.
{rubric}
1. THE CUT: choose recommended_end_timestamp = the END time of one line below, between
   {min_end} and {max_end} (20 to 60 seconds after the start). Cut at the most
   unresolved moment: a question with no answer, a threat, a reveal, a decision not yet
   made. Never cut right after the conflict is resolved.
2. GRADE 1-10 with these anchors:
   hook_intensity_score: 9-10 physical shock, public humiliation, identity reveal or
     betrayal in the first 3s; 5-6 tense talk but no shock; 1-3 calm/establishing.
   cliffhanger_intensity_score: 9-10 viewer NEEDS the next second (proposal before the
     answer, a threat, a reveal); 5-6 some tension left; 1-3 resolved, calm or mid-sentence.
   middle_story_engagement_score: 9-10 something new every few seconds, escalating;
     1-3 long repetitive or slow stretch a viewer would swipe away from.
3. AMNESIA TEST: the viewer has NEVER seen this show and knows only what is on screen
   in this clip. Speaker labels like "BLOND NOBLEMAN" are what the viewer sees.
   is_self_understandable = false if the clip depends on backstory, unexplained names
   or earlier events; say what is missing in missing_context_explanation ("" if none).
4. is_valid_promo = true only if the hook lands in the first 3 seconds, the ending is
   unresolved, and the amnesia test passes.
5. Write: title (2-5 words), hook_text (what the viewer sees/hears in the first 3s),
   cliffhanger_text (the final moment), why_it_works (ONE line).
6. hook_archetype and cliffhanger_archetype: {archetype_rule}

EPISODE {episode} - clip window
{script}
"""


def judge(llm: LLM, clip: dict) -> Optional[PromoJudgeEvaluation]:
    s = clip["start"]
    prompt = JUDGE_PROMPT.format(
        window=clip["window_end"] - s, start=fmt(s), hook=clip["hook_description"],
        rubric=RUBRIC["text"], archetype_rule=_archetype_rule(),
        min_end=fmt(s + 20), max_end=fmt(min(s + 60, clip["window_end"])),
        episode=clip["episode"], script=render(clip["events"]))
    return llm.structured(prompt, PromoJudgeEvaluation, f"judge-{clip['clip_id']}",
                          mock=lambda: _mock_judge(clip))


# =============================================================================
# Phase 5 - Ranker (one call for the whole show)
# =============================================================================

RANK_PROMPT = """You decide which promo clips a microdrama app posts FIRST on TikTok,
Reels and Shorts to pull in NEW viewers who have never seen the show.

Order ALL {n} clips below from post-first to post-last. Prefer: the strongest first
3 seconds, an ending that forces a click, clips a stranger fully understands, early-story
setups over late twists that spoil the ending, and variety (do not stack near-identical
moments next to each other). Give each a one-line reason. Use the exact clip_id values.

CLIPS
{clips}
"""


def ranker(llm: LLM, clips: list[dict]) -> list[RankedClip]:
    rows = [json.dumps({
        "clip_id": c["clip_id"], "episode": c["episode"], "seconds": round(c["duration"], 1),
        "title": c["title"], "hook": c["hook_text"], "cliffhanger": c["cliffhanger_text"],
        "why": c["why_it_works"], "scores": [c["hook_intensity_score"],
                                             c["cliffhanger_intensity_score"],
                                             c["middle_story_engagement_score"]]})
        for c in clips]
    prompt = RANK_PROMPT.format(n=len(clips), clips="\n".join(rows))
    res = llm.structured(prompt, Ranking, "rank", mock=lambda: _mock_rank(clips))
    return res.ranking if res else []


# =============================================================================
# Rules mode (provider: rules; "mock" is the same thing): no LLM, no API key.
# Reads the script's own signals - bracketed emotions, action verbs, sound cues,
# "!", "?" and trailing "..." - to find hooks and cliffhangers, and flags names a
# stranger never sees. Used for the submitted run and as a no-AI baseline.
# =============================================================================

HOT_EMOTIONS = ("shock", "furious", "scream", "outrag", "hysteric", "gasp", "betray",
                "devastat", "venom", "menac", "threat", "cruel", "vengeful", "desperate",
                "horrif", "accus", "defian", "deadly", "brutal", "panic", "terror")
HOOK_KINDS = [   # label, word stems (matched at the start of a word)
    ("physical attack", r"slap|strik|punch|shove|grab|seiz|drag|collaps|smash|shatter|"
                        r"slam|chok|struggl|scuffl|blocks the|charges"),
    ("proposal", r"kneel|one knee|ring\b|propos|marry me|accept my hand|kiss"),
    ("reveal", r"reveal|who are you|you already know|truth|secret|alive|rightful|"
               r"is my son|imposter|unmask|you cannot be him"),
    ("public humiliation", r"humiliat|mock|scorn|gloat|smug|sneer|disgrac|embarrass|"
                           r"discarded|worthless|plainness|low value|tolerate"),
    ("betrayal line", r"betray|you knew|lied|stole|marrying me|changed my mind|"
                      r"never going to marry|swore"),
    ("outburst", r"scream|shout|furious|hysteric|outrag|enough|guards"),
]
_KIND_RE = [(k, re.compile(r"\b(?:" + pat + ")", re.I)) for k, pat in HOOK_KINDS]
_THREAT = re.compile(r"\b(?:vengeful|menac|threat|settle|pay for|regret|you will|"
                     r"retribution|warn)", re.I)
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")
_NOT_NAMES = {"The", "You", "Your", "And", "But", "What", "Why", "How", "Who", "Where",
              "When", "Yes", "Please", "Look", "Let", "Now", "Then", "Not", "This", "That",
              "She", "They", "Enough", "Guards", "Mother", "Father", "Lady", "Lord", "Sir",
              "Duke", "Duchess", "God", "Because", "Are", "Take", "Stand", "Don", "Can",
              "Brother", "Sister", "Your Grace", "Grace", "Majesty", "Highness", "Madam"}


def _blob(e: dict) -> str:
    return f"{e['emotion']} {e['text']}"


def _kind(e: dict) -> str:
    return next((k for k, rx in _KIND_RE if rx.search(_blob(e))), "")


def _heat(e: dict) -> int:
    t = _blob(e).lower()
    h = sum(w in t for w in HOT_EMOTIONS) * 2 + (3 if _kind(e) else 0)
    return h + min(2, e["text"].count("!")) + (e["type"] == "sfx")


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _line(e: dict) -> str:
    if e["type"] == "dialogue":
        return f'{string.capwords(e["speaker"].lower())}: "{e["text"]}"'
    return e["text"]


def _mock_hooks(ep: dict, n: int) -> HookList:
    latest = ep["duration"] - 20
    pool = sorted((e for e in ep["events"] if e["type"] in ("dialogue", "action", "sfx")
                   and e["start"] <= latest), key=lambda e: (-_heat(e), e["start"]))
    picked: list[dict] = []
    for e in pool:
        if _heat(e) >= 3 and all(abs(e["start"] - p["start"]) >= 10 for p in picked):
            picked.append(e)
        if len(picked) == n:
            break
    return HookList(hooks=[Hook(start_timestamp=fmt(e["start"]),
                                hook_type=_kind(e) or "emotional spike",
                                hook_description=_line(e)[:160]) for e in picked])


def _cliff_score(e: dict) -> int:
    t = e["text"].rstrip()
    return min(10, 3 + 3 * ("?" in t) + 2 * t.endswith("...") + 2 * bool(_THREAT.search(_blob(e)))
               + min(3, _heat(e)))


def _cliff_kind(e: dict) -> str:
    t = e["text"].rstrip()
    if "?" in t:
        return "unanswered question"
    if _THREAT.search(_blob(e)):
        return "threat"
    if t.endswith("..."):
        return "unfinished line"
    return _kind(e) or "tense beat"


def _unseen_names(clip: dict, end: float) -> list[str]:
    """Names spoken in the clip that the viewer never sees on screen or as a speaker."""
    evs = [e for e in clip["events"] if e["start"] < end]
    seen = " ".join([(e["speaker"] or "") for e in evs] +
                    [e["text"] for e in evs if e["type"] != "dialogue"]).lower()
    said = {m for e in evs if e["type"] == "dialogue" for m in _NAME.findall(e["text"])}
    return sorted(n for n in said - _NOT_NAMES if n.lower() not in seen)


def _mock_judge(clip: dict) -> PromoJudgeEvaluation:
    s, hi = clip["start"], min(clip["start"] + 60, clip["window_end"])
    ends = [e for e in clip["events"] if e["type"] == "dialogue" and s + 20 <= e["end"] <= hi]
    if not ends:
        ends = [e for e in clip["events"] if s + 20 <= e["end"] <= hi] or clip["events"][-1:]
    end_ev = max(ends, key=lambda e: (_cliff_score(e), e["end"]))
    first = [e for e in clip["events"] if e["start"] <= s + 3]
    mid = [e for e in clip["events"] if s + 3 < e["start"] < end_ev["start"]]
    span = max(1.0, end_ev["start"] - s - 3)
    hook = min(10, 3 + sum(map(_heat, first)))
    cliff = _cliff_score(end_ev)
    middle = min(10, 2 + round(sum(_heat(e) >= 2 for e in mid) / span * 30))
    missing = _unseen_names(clip, end_ev["end"])
    hook_kind = clip["hook_type"] or "emotional spike"
    cliff_kind = _cliff_kind(end_ev)
    words = re.sub(r"[^\w' ]", "", clip["hook_description"].split(":", 1)[-1]).split()
    return PromoJudgeEvaluation(
        is_valid_promo=hook >= 5 and cliff >= 5 and not missing,
        recommended_end_timestamp=fmt(end_ev["end"]),
        hook_intensity_score=hook, cliffhanger_intensity_score=cliff,
        middle_story_engagement_score=middle, is_self_understandable=not missing,
        missing_context_explanation=(f"mentions {', '.join(missing)}, never shown or "
                                     "introduced in the clip") if missing else "",
        title=" ".join(words[:5]) or hook_kind.title(),
        hook_text=clip["hook_description"], cliffhanger_text=_line(end_ev)[:200],
        why_it_works=f"Opens on {_a(hook_kind)} in the first 3 seconds and cuts on "
                     f"{_a(cliff_kind)}.",
        hook_archetype=hook_kind, cliffhanger_archetype=cliff_kind)


def _mock_rank(clips: list[dict]) -> Ranking:
    """Best score first, but a clip that overlaps one already placed waits until the end,
    so the top of the list is not the same moment twice."""
    by_score = sorted(clips, key=lambda c: (-(c["hook_intensity_score"]
                                              + c["cliffhanger_intensity_score"]),
                                            -c["middle_story_engagement_score"], c["episode"]))
    first, later, placed = [], [], set()
    for c in by_score:
        (later if placed & set(c.get("overlaps_with", [])) else first).append(c)
        placed.add(c["clip_id"])

    def why(c, dup):
        r = (f"hook {c['hook_intensity_score']}/10, cliffhanger "
             f"{c['cliffhanger_intensity_score']}/10")
        if c["episode"] <= 10:
            r += "; early episode, needs no backstory"
        return r + ("; overlaps a higher pick, so placed later" if dup else "")
    return Ranking(ranking=[RankedClip(clip_id=c["clip_id"], reason=why(c, c in later))
                            for c in first + later])
