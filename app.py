"""
Promo Extractor - Streamlit app.

    streamlit run app.py

Open results  : upload a promos.json from a run (and, optionally, your labelled
                hand_check.csv) to browse the posting queue.
Run on scripts: upload a zip of episode .txt scripts and choose how the judging is done:
                - copy-paste with ChatGPT / Claude (LLM, no API key),
                - instant rules (no AI),
                - an API key (OpenAI, Claude, Gemini).
Scripts are processed in a temporary folder for this session only and are not stored.
"""

from __future__ import annotations

import csv
import html
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Promo Extractor", page_icon="🎬", layout="wide")
HERE = Path(__file__).parent
REPO_URL = "https://github.com/ShubNoob/promo-extractor"

# ----------------------------------------------------------------------------- look
# Kuku black and orange. On every clip strip the hook is the orange tick and the cut the black one.
INK, MUTED, RULE, PAPER = "#121212", "#5A5A5A", "#E4E4E4", "#FAFAFA"
ORANGE, ORANGE_TEXT, TINT, SPAN = "#FF6A13", "#C2410C", "#FFEBDD", "#FFC9A6"
DROP = "#8A8A8A"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
.stApp {{ background: {PAPER}; color: {INK}; font-family: 'Inter', system-ui, sans-serif; }}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea {{
  font-family: 'Inter', system-ui, sans-serif; color: {INK}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1100px; }}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
  background: {ORANGE} !important; border-color: {ORANGE} !important; color: {INK} !important; font-weight: 700; }}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{
  background: #F25A00 !important; border-color: #F25A00 !important; }}
button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {{ color: {INK} !important; font-weight: 700; }}
.stTabs [aria-selected="true"] p {{ font-weight: 700; }}
.stApp a {{ color: {ORANGE_TEXT}; font-weight: 600; }}
.band {{ background: {INK}; color: #fff; border-radius: 16px; padding: 1.6rem 1.8rem 1.2rem; margin: 0 0 1.2rem; }}
.band .kicker {{ color: {ORANGE}; font-weight: 600; font-size: 0.9rem; margin: 0 0 0.35rem; }}
.band .bt {{ font-weight: 900; font-size: 2.3rem; letter-spacing: -0.03em; line-height: 1.08; color: #fff; }}
.band .bs {{ color: #D9D9D9; font-size: 1.04rem; line-height: 1.55; max-width: 62ch; margin: 0.6rem 0 0; }}
.band .bs b {{ color: #fff; }}
.band .mini {{ position: relative; height: 16px; max-width: 560px; margin: 1.1rem 0 0.1rem; }}
.band .mini .tr {{ position: absolute; left: 0; right: 0; top: 6px; height: 4px; border-radius: 2px; background: #333; }}
.band .mini .sp {{ position: absolute; left: 34%; width: 27%; top: 4px; height: 8px; border-radius: 2px; background: #F2F2F2; }}
.band .mini .tk {{ position: absolute; top: 0; width: 4px; height: 16px; border-radius: 1px; transform: translateX(-50%); }}
.lede {{ font-size: 1.04rem; color: {MUTED}; max-width: 64ch; line-height: 1.55; margin: 0 0 1rem; }}
.summary {{ font-size: 1.02rem; max-width: 72ch; line-height: 1.55; margin: 0 0 1rem; }}
.summary b {{ font-weight: 700; }}
.clip {{ display: grid; grid-template-columns: 3.2rem 1fr; gap: 0.9rem; padding: 1.1rem 0;
  border-top: 1px solid {RULE}; }}
.clip .rank {{ font-size: 1.9rem; font-weight: 900; line-height: 1; color: {INK};
  font-variant-numeric: tabular-nums; text-align: right; padding-top: 0.1rem; }}
.clip .rank.none {{ color: {RULE}; }}
.clip .head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.35rem 1rem; }}
.clip .title {{ font-weight: 800; font-size: 1.08rem; }}
.clip .when {{ color: {MUTED}; font-variant-numeric: tabular-nums; }}
.clip .lines {{ margin-top: 0.35rem; max-width: 76ch; line-height: 1.5; }}
.clip .lines p {{ margin: 0.25rem 0; }}
.clip .tag {{ font-weight: 800; margin-right: 0.35rem; }}
.clip .tag.hook {{ color: {ORANGE_TEXT}; }}
.clip .tag.cut {{ color: {INK}; }}
.clip .why {{ color: {MUTED}; }}
.clip .facts {{ display: flex; flex-wrap: wrap; gap: 0.3rem 1.1rem; margin-top: 0.45rem;
  font-size: 0.88rem; color: {MUTED}; font-variant-numeric: tabular-nums; }}
.clip .note {{ font-size: 0.88rem; margin-top: 0.35rem; }}
.clip .note.trim {{ color: {ORANGE_TEXT}; }}
.clip .note.drop {{ color: {MUTED}; }}
.verdict {{ font-size: 0.82rem; font-weight: 700; padding: 0.08rem 0.55rem; border-radius: 999px; }}
.verdict.yes {{ background: {TINT}; color: {ORANGE_TEXT}; }}
.verdict.no {{ background: #EFEFEF; color: {MUTED}; }}
.strip {{ position: relative; height: 22px; max-width: 640px; margin: 0.6rem 0 0.15rem; }}
.strip .track {{ position: absolute; left: 0; right: 0; top: 9px; height: 4px; border-radius: 2px; background: {RULE}; }}
.strip .span {{ position: absolute; top: 6px; height: 10px; min-width: 3px; border-radius: 2px; background: {SPAN}; }}
.strip .tick {{ position: absolute; top: 1px; width: 3px; height: 20px; border-radius: 1px; transform: translateX(-50%); }}
.strip .tick.hook {{ background: {ORANGE}; }}
.strip .tick.cut {{ background: {INK}; }}
.strip-ends {{ display: flex; justify-content: space-between; max-width: 640px; font-size: 0.75rem;
  color: {MUTED}; font-variant-numeric: tabular-nums; }}
.empty {{ color: {MUTED}; padding: 1.2rem 0; border-top: 1px solid {RULE}; }}
</style>
""", unsafe_allow_html=True)


def band(kicker: str, title: str, sub: str = "", strip: bool = False):
    """Black header band, as on the 2-page submission."""
    mini = ("<div class='mini' aria-hidden='true'><div class='tr'></div><div class='sp'></div>"
            f"<div class='tk' style='left:34.5%;background:{ORANGE}'></div>"
            "<div class='tk' style='left:61%;background:#fff'></div></div>") if strip else ""
    st.markdown(f"<div class='band'><p class='kicker'>{kicker}</p>"
                f"<div class='bt' role='heading' aria-level='1'>{title}</div>"
                f"{f'<p class=bs>{sub}</p>' if sub else ''}{mini}</div>", unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x or ""))


def tc(sec: float) -> str:
    m, s = divmod(max(0.0, float(sec)), 60)
    return f"{int(m):02d}:{s:05.2f}"


# ----------------------------------------------------------------------------- state helpers

def load_result(data: dict, files: dict | None = None):
    st.session_state["result"] = data
    st.session_state["files"] = files or {}


def reset():
    for k in ("result", "files", "hand", "manual"):
        st.session_state.pop(k, None)


def read_hand_check(raw: bytes) -> dict:
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    out = {}
    for r in rows:
        v = (r.get("human_good") or "").strip().lower()
        if v[:1] in ("y", "n"):
            out[r["clip_id"]] = {"good": v.startswith("y"), "reason": r.get("human_reason", "")}
    return out


def run_pipeline(zip_bytes: bytes, name: str, provider: str, extra: list[str], env_add: dict):
    work = Path(tempfile.mkdtemp())
    zpath = work / "scripts.zip"
    zpath.write_bytes(zip_bytes)
    cmd = [sys.executable, str(HERE / "run.py"), str(zpath), "--out", str(work / "out"),
           "--provider", provider, "--show", Path(name).stem] + extra
    log, lines = st.empty(), []
    with st.spinner("Finding promos..."):
        p = subprocess.Popen(cmd, cwd=HERE, env={**os.environ, **env_add}, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            lines.append(line.rstrip())
            log.code("\n".join(lines[-10:]), language=None)
        p.wait()
    out = work / "out"
    if p.returncode == 0 and (out / "promos.json").exists():
        log.empty()
        load_result(json.loads((out / "promos.json").read_text()),
                    {f.name: f.read_bytes() for f in sorted(out.iterdir())})
        st.rerun()
    st.error("The run stopped with an error. The last lines of its log are shown above.")


# ----------------------------------------------------------------------------- manual (copy-paste) mode

def manual_prepare(m: dict) -> bool:
    """Run the pipeline in manual mode: reuses saved answers, writes the next round."""
    cmd = [sys.executable, str(HERE / "run.py"), m["zip"], "--provider", "manual",
           "--model", m["label"], "--manual-dir", m["dir"], "--hooks", str(m["hooks"]),
           "--max-chars", str(m["max_chars"]), "--out", m["out"], "--show", m["show"]]
    if m["episodes"]:
        cmd += ["--episodes", m["episodes"]]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        m["error"] = (r.stdout + r.stderr)[-1500:]
        return False
    m["tried"], m["cur"] = [], None
    return True


def manual_screen(m: dict):
    import manual
    allb = manual.batches(m["dir"])
    if not allb:                                   # nothing pending: final results exist
        out = Path(m["out"])
        load_result(json.loads((out / "promos.json").read_text()),
                    {f.name: f.read_bytes() for f in sorted(out.iterdir())})
        st.session_state.pop("manual", None)
        st.rerun()
    todo = [b for b in allb if not b["complete"] and b["name"] not in m["tried"]]
    if not todo:                                   # round handled -> next round
        with st.spinner("Preparing the next round..."):
            if manual_prepare(m):
                st.rerun()
        st.error("Preparing the next round failed.")
        st.code(m.get("error", ""), language=None)
        return
    b = next((x for x in todo if x["name"] == m["cur"]), todo[0])
    m["cur"] = b["name"]
    done = sum(x["complete"] or x["name"] in m["tried"] for x in allb)

    band("Copy-paste run with " + esc(m["label"]), esc(manual.round_name([b])))
    st.markdown(f"<p class='lede'>Prompt {allb.index(b) + 1} of {len(allb)} in this round, "
                f"{done} done. It holds {len(b['tags'])} tasks ({b['chars']:,} characters). "
                f"Copy it with the button at its top right, paste it into a new chat with "
                f"{esc(m['label'])}, wait for the full reply, then paste the reply below.</p>",
                unsafe_allow_html=True)
    with st.container(height=260):
        st.code(b["text"], language=None)
    reply = st.text_area("The model's whole reply", key=f"reply-{b['name']}-{len(m['tried'])}",
                         height=180, placeholder="Paste the reply here, including any text around the JSON")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Save reply", type="primary", use_container_width=True):
        saved, problems = manual.save_answer(m["dir"], b["name"], reply)
        if not saved:
            st.error("Nothing was saved: " + "; ".join(problems))
            return
        m["tried"].append(b["name"])
        m["flash"] = (f"Saved {len(saved)} of {len(b['tags'])} answers."
                      + (f" {len(problems)} missing or malformed, asked again next round."
                         if problems else ""))
        st.rerun()
    if c2.button("Skip for now", use_container_width=True) and len(todo) > 1:
        names = [x["name"] for x in todo]
        m["cur"] = names[(names.index(b["name"]) + 1) % len(names)]
        st.rerun()
    if m.get("flash"):
        st.success(m.pop("flash"))
    st.caption("Saved answers are checked against the expected format, and every timestamp is "
               "checked against the script, before anything reaches the results.")


# ----------------------------------------------------------------------------- landing

def landing():
    band("Promo Extractor for microdramas", "Find the promo inside every episode",
         "Upload a show's screenplays and get back ranked 20 to 60 second promo clips: a hook in "
         "the first 3 seconds, a cut on a cliffhanger, clear to someone who has never seen the "
         "show, with exact timestamps to cut from.", strip=True)
    open_tab, run_tab = st.tabs(["Open results", "Run on scripts"])

    with open_tab:
        st.write("Open the `promos.json` from a run, for example the Colab notebook.")
        up = st.file_uploader("promos.json", type="json", key="up-json")
        hc = st.file_uploader("Your labelled hand_check.csv (optional)", type="csv", key="up-hc")
        if up is not None:
            if hc is not None:
                st.session_state["hand"] = read_hand_check(hc.getvalue())
            load_result(json.load(up))
            st.rerun()

    with run_tab:
        zip_up = st.file_uploader("Episode scripts, as a .zip of .txt files", type="zip", key="up-zip")
        engine = st.radio(
            "Who judges the clips", ["copy", "rules", "api"], horizontal=False,
            format_func=lambda e: {
                "copy": "ChatGPT or Claude, by copy-paste (LLM, no API key)",
                "rules": "Instant rules (no AI, seconds)",
                "api": "An API key (OpenAI, Claude or Gemini)"}[e])
        c1, c2 = st.columns(2)
        hooks = c1.number_input("Hooks per episode", 1, 5, 3)
        episodes = c2.text_input("Episodes", "", placeholder="all, or e.g. 1-10")
        if engine == "copy":
            c3, c4 = st.columns(2)
            label = c3.text_input("Chat model you will paste into", "ChatGPT (web)")
            max_chars = c4.number_input("Characters per prompt", 10000, 150000, 40000, step=10000,
                                        help="40,000 suits ChatGPT; Claude.ai takes 90,000+.")
            st.caption("The app packs the model's tasks into a few large prompts. You paste each into "
                       "the chat and paste the reply back; three rounds: find hooks, judge clips, rank.")
        elif engine == "api":
            c3, c4, c5 = st.columns(3)
            provider = c3.selectbox("Provider", ["openai", "anthropic", "gemini"],
                                    format_func=lambda p: {"openai": "OpenAI", "anthropic": "Claude",
                                                           "gemini": "Gemini"}[p])
            model = c4.text_input("Model", {"openai": "gpt-5-mini", "anthropic": "claude-sonnet-5-5",
                                            "gemini": "gemini-flash-latest"}[provider])
            key_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                       "gemini": "GEMINI_API_KEY"}[provider]
            key = c5.text_input("API key", type="password")

        if st.button("Find promos", type="primary", disabled=zip_up is None):
            extra = ["--hooks", str(hooks)] + (["--episodes", episodes.strip()] if episodes.strip() else [])
            if engine == "rules":
                run_pipeline(zip_up.getvalue(), zip_up.name, "rules", extra, {})
            elif engine == "api":
                if not key:
                    st.error("Add an API key, or choose the copy-paste option, which needs none.")
                else:
                    run_pipeline(zip_up.getvalue(), zip_up.name, provider,
                                 extra + ["--model", model], {key_env: key})
            else:
                work = Path(tempfile.mkdtemp())
                (work / "scripts.zip").write_bytes(zip_up.getvalue())
                m = {"dir": str(work / "manual"), "zip": str(work / "scripts.zip"),
                     "out": str(work / "out"), "label": label or "chat model",
                     "hooks": int(hooks), "episodes": episodes.strip(),
                     "max_chars": int(max_chars), "show": Path(zip_up.name).stem}
                with st.spinner("Preparing round 1..."):
                    ok = manual_prepare(m)
                if ok:
                    st.session_state["manual"] = m
                    st.rerun()
                st.error("Preparing the prompts failed.")
                st.code(m.get("error", ""), language=None)
        if zip_up is None:
            st.caption("Scripts are processed in a temporary folder for this session and not stored.")
    st.markdown(f"<p class='lede' style='margin-top:2rem'>Method, code and notebook: "
                f"<a href='{REPO_URL}'>{REPO_URL.split('//')[1]}</a></p>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- results

def strip_html(start: float, end: float, hook_at: float, ep_len: float, known: bool) -> str:
    """The episode as a track; the clip as a solid span; the hook (rose) and the cut (cyan) as ticks."""
    pct = lambda t: max(0.0, min(100.0, t / ep_len * 100))
    s, e, h = pct(start), pct(end), pct(start + hook_at)
    ends = (f"<div class='strip-ends'><span>00:00</span><span>{tc(ep_len)}</span></div>" if known else "")
    return (f"<div class='strip' role='img' aria-label='Clip from {tc(start)} to {tc(end)} of the episode'>"
            f"<div class='track'></div><div class='span' style='left:{s:.2f}%;width:{e - s:.2f}%'></div>"
            f"<div class='tick hook' style='left:{h:.2f}%'></div>"
            f"<div class='tick cut' style='left:{e:.2f}%'></div></div>{ends}")


def card(r: dict, ep_len: float, known: bool, hand: dict, show_rank: bool = True):
    s = r["segments"][0]
    rank = r.get("rank") if show_rank else None
    v = hand.get(r["clip_id"])
    verdict = ""
    if v:
        verdict = (f"<span class='verdict {'yes' if v['good'] else 'no'}'>You: "
                   f"{'good' if v['good'] else esc(v['reason'] or 'not good')}</span>")
    notes = ""
    if r["tier"] == 2:
        notes += f"<div class='note trim'>{esc(r['tier_reason'])}</div>"
    if r["tier"] == 3:
        notes += f"<div class='note drop'>Rejected: {esc(r['tier_reason'])}</div>"
    if r.get("timing_note"):
        notes += f"<div class='note'>Timing fixed by the checks: {esc(r['timing_note'])}</div>"
    st.markdown(f"""
<div class="clip">
  <div class="rank {'none' if not rank else ''}">{rank or '·'}</div>
  <div>
    <div class="head"><span class="title">{esc(r['title'])}</span>
      <span class="when">Episode&nbsp;{s['episode']}, {s['start_ts']} to {s['end_ts']}, {r['duration']:.0f}&nbsp;s</span>
      {verdict}</div>
    {strip_html(r['start'], r['end'], r.get('hook_at_second', 0), ep_len, known)}
    <div class="lines">
      <p><span class="tag hook">Hook</span>{esc(r['hook_text'])}</p>
      <p><span class="tag cut">Cut</span>{esc(r['cliffhanger_text'])}</p>
      <p class="why">{esc(r['why_it_works'])}</p>
    </div>
    <div class="facts"><span>Hook&nbsp;{r['hook_intensity_score']}/10</span>
      <span>Cliffhanger {r['cliffhanger_intensity_score']}/10</span>
      <span>Middle {r['middle_story_engagement_score']}/10</span>
      <span>{esc(r['clip_id'])}</span>
      {f"<span>Overlaps {esc(', '.join(r['overlaps_with']))}</span>" if r.get('overlaps_with') else ''}</div>
    {notes}
  </div>
</div>""", unsafe_allow_html=True)


def results(data: dict):
    meta, promos = data["meta"], data["promos"]
    rejected = data.get("rejected_hooks", [])
    hand = st.session_state.get("hand", {})
    files = st.session_state.get("files", {})
    tiers = {str(k): v for k, v in meta.get("tier_counts", {}).items()}
    t1 = sorted([r for r in promos if r["tier"] == 1], key=lambda r: r.get("rank") or 10 ** 6)
    t2 = [r for r in promos if r["tier"] == 2]
    t3 = [r for r in promos if r["tier"] == 3]

    # episode lengths: from the run log, else estimated from the latest clip end in that episode
    known = {int(k): float(v) for k, v in meta.get("episode_durations", {}).items()}
    est = {}
    for r in promos:
        est[r["episode"]] = max(est.get(r["episode"], 0.0), r["end"])
    ep_len = lambda e: (known.get(e) or est.get(e, 60.0) * 1.05, e in known)

    with st.sidebar:
        st.markdown(f"**{esc(meta.get('show', 'Show'))}**")
        st.caption(f"Judged by {meta.get('model', '')}")
        hc = st.file_uploader("Add your labelled hand_check.csv", type="csv", key="side-hc")
        if hc is not None:
            st.session_state["hand"] = hand = read_hand_check(hc.getvalue())
        if st.button("Open another run", use_container_width=True):
            reset()
            st.rerun()

    band(f"Judged by {esc(meta.get('model', ''))}", f"Promos for {esc(meta.get('show', 'this show'))}")
    agree = ""
    if hand:
        good = sum(v["good"] for v in hand.values())
        agree = f" In your hand check you agreed with <b>{good} of {len(hand)}</b> top picks."
    st.markdown(
        f"<p class='summary'>{meta.get('episodes', 0)} episodes, {meta.get('hooks_found', 0)} hooks "
        f"found, {meta.get('clips_judged', 0)} clips judged. <b>{tiers.get('1', 0)} are ready to "
        f"post</b>, {tiers.get('2', 0)} need a trim and {tiers.get('3', 0)} were rejected; "
        f"the checks threw out {meta.get('rejected_before_judging', 0)} more before judging and "
        f"corrected {meta.get('end_timestamps_adjusted', 0)} end times.{agree}</p>",
        unsafe_allow_html=True)
    if meta.get("stopped_early"):
        st.warning(f"Partial run: {meta['stopped_early']}")
    if meta.get("pending_manual_tasks"):
        st.warning(f"{meta['pending_manual_tasks']} tasks were still unanswered when this was saved.")

    queue, trim, drop, table, eps = st.tabs([
        f"Posting queue ({len(t1)})", f"Needs a trim ({len(t2)})",
        f"Rejected ({len(t3) + len(rejected)})", "All clips", "By episode"])

    with queue:
        c1, c2, c3 = st.columns([2, 2, 1.4])
        q = c1.text_input("Search hooks and cuts", placeholder="e.g. guards, ring, poison")
        lo_ep = min((r["episode"] for r in promos), default=1)
        hi_ep = max((r["episode"] for r in promos), default=1)
        rng = c2.slider("Episodes", lo_ep, hi_ep, (lo_ep, hi_ep)) if hi_ep > lo_ep else (lo_ep, hi_ep)
        dedupe = c3.toggle("Hide repeats", value=False,
                           help="Hide a clip that overlaps a higher-ranked one from the same episode.")
        shown, kept = [], set()
        for r in t1:
            if not rng[0] <= r["episode"] <= rng[1]:
                continue
            text = f"{r['title']} {r['hook_text']} {r['cliffhanger_text']}".lower()
            if q and q.lower() not in text:
                continue
            if dedupe and kept & set(r.get("overlaps_with", [])):
                continue
            kept.add(r["clip_id"])
            shown.append(r)
        st.caption(f"Showing {len(shown)} of {len(t1)}, in the order to post them. "
                   "On each strip, the bar is the clip inside its episode, the orange tick is the hook "
                   "and the black tick is the cut.")
        if not shown:
            st.markdown("<div class='empty'>No clip matches these filters.</div>", unsafe_allow_html=True)
        for r in shown[:60]:
            card(r, *ep_len(r["episode"]), hand)
        if len(shown) > 60:
            st.caption(f"{len(shown) - 60} more in the All clips tab.")

    with trim:
        if not t2:
            st.markdown("<div class='empty'>No clip was marked as needing a trim.</div>",
                        unsafe_allow_html=True)
        for r in t2:
            card(r, *ep_len(r["episode"]), hand, show_rank=False)

    with drop:
        for r in t3:
            card(r, *ep_len(r["episode"]), hand, show_rank=False)
        if rejected:
            st.subheader("Thrown out before judging")
            st.dataframe(pd.DataFrame(rejected)[["clip_id", "stage", "reason"]],
                         hide_index=True, use_container_width=True)

    with table:
        df = pd.DataFrame([{
            "Rank": r.get("rank"), "Tier": r["tier"], "Clip": r["clip_id"], "Episode": r["episode"],
            "Start": r["segments"][0]["start_ts"], "End": r["segments"][0]["end_ts"],
            "Seconds": round(r["duration"], 1), "Hook": r["hook_intensity_score"],
            "Cliffhanger": r["cliffhanger_intensity_score"], "Middle": r["middle_story_engagement_score"],
            "Title": r["title"], "You": ("good" if hand[r["clip_id"]]["good"] else "not good")
            if r["clip_id"] in hand else ""} for r in promos])
        st.dataframe(df, hide_index=True, use_container_width=True, height=480)

    with eps:
        counts = pd.DataFrame([{"Episode": r["episode"], "Tier": {1: "Ready", 2: "Trim", 3: "Rejected"}[r["tier"]]}
                               for r in promos])
        if not counts.empty:
            pivot = counts.value_counts().unstack(fill_value=0).reindex(
                columns=["Ready", "Trim", "Rejected"], fill_value=0).sort_index()
            st.bar_chart(pivot, color=[ORANGE, SPAN, '#B8B8B8'], height=320)
            st.caption("Judged clips per episode, by outcome.")

    st.divider()
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("promos.json", json.dumps(data, indent=2), "promos.json", use_container_width=True)
    d2.download_button("promos.csv", files.get("promos.csv") or df.to_csv(index=False), "promos.csv",
                       use_container_width=True)
    if "shortlist.md" in files:
        d3.download_button("shortlist.md", files["shortlist.md"], "shortlist.md", use_container_width=True)
    if "hand_check.csv" in files:
        d4.download_button("hand_check.csv", files["hand_check.csv"], "hand_check.csv",
                           use_container_width=True)


# ----------------------------------------------------------------------------- page

if "manual" in st.session_state:
    with st.sidebar:
        st.markdown("**Copy-paste run in progress**")
        st.caption("Your answers are kept while this tab stays open.")
        if st.button("Cancel this run", use_container_width=True):
            reset()
            st.rerun()
    manual_screen(st.session_state["manual"])
elif st.session_state.get("result"):
    results(st.session_state["result"])
else:
    landing()
