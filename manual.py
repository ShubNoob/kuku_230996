"""
Manual LLM mode: you are the API.

The pipeline is the same as LLM mode, but instead of calling a model it writes the model's
tasks into a few large prompts ("batches"). You paste each batch into ChatGPT, Claude or
Gemini in the browser, paste the reply back, and Python checks every answer (and, as in
every mode, every timestamp) before the next round is prepared:

    round 1  find hooks     (1 task per episode)
    round 2  judge clips    (1 task per clip)
    round 3  rank Tier 1    (1 task)

Colab: `manual.colab_ui(ZIP)` shows each batch with a box to paste the reply into.
Command line:
    python run.py scripts.zip --provider manual           # prepare the next round
    python manual.py status                               # what is answered / pending
    python manual.py answer batch_01 reply.txt            # save a pasted reply
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import typing
from pathlib import Path

from pydantic import BaseModel, ValidationError

from llm_nodes import HookList, PromoJudgeEvaluation, Ranking

SCHEMAS = {m.__name__: m for m in (HookList, PromoJudgeEvaluation, Ranking)}
ROUND_NAMES = {"hooks": "Round 1 of 3: find hooks", "judge": "Round 2 of 3: judge clips",
               "rank": "Round 3 of 3: rank the best clips"}

HEADER = """You will complete {n} separate tasks for a promo-clip tool. Each task sits between
"=== TASK <id> ===" and "=== END TASK <id> ===". Do every task, each one on its own,
following its own instructions exactly (copy timestamps exactly as they appear).

Reply with ONLY one JSON object and nothing else, with one key per task id:
{{{keys}}}

Each answer must have exactly this shape (replace every <...> with the real value;
numbers and true/false go without quotes):
{shape}
"""


# ----------------------------------------------------------------------------- shapes

def _shape(t, name: str = ""):
    if typing.get_origin(t) is list:
        return [_shape(typing.get_args(t)[0])]
    if isinstance(t, type) and issubclass(t, BaseModel):
        return {k: _shape(f.annotation, k) for k, f in t.model_fields.items()}
    if t is bool:
        return "<true or false>"
    if t is int:
        return "<integer 1-10>"
    if t is float:
        return "<number>"
    if "timestamp" in name:
        return "<MM:SS.ss copied from the script>"
    if name in ("clip_id",):
        return "<clip_id exactly as given>"
    return "<text>"


def shape_text(schema: type[BaseModel]) -> str:
    return json.dumps(_shape(schema), indent=1)


# ----------------------------------------------------------------------------- batches

def _dirs(manual_dir) -> tuple[Path, Path]:
    root = Path(manual_dir)
    (root / "answers").mkdir(parents=True, exist_ok=True)
    (root / "batches").mkdir(parents=True, exist_ok=True)
    return root / "answers", root / "batches"


def write_batches(pending: list[dict], manual_dir, max_chars: int = 40000) -> list[dict]:
    """Group pending tasks (same schema only) into prompts of at most ~max_chars."""
    _, bdir = _dirs(manual_dir)
    for old in bdir.glob("*"):
        old.unlink()
    groups: list[list[dict]] = []
    for task in pending:
        g = groups[-1] if groups else None
        size = sum(len(t["prompt"]) + 80 for t in g) if g else 0
        if g and g[0]["schema"] == task["schema"] and size + len(task["prompt"]) < max_chars:
            g.append(task)
        else:
            groups.append([task])
    index = {}
    for i, g in enumerate(groups, 1):
        name = f"batch_{i:02d}"
        schema = SCHEMAS[g[0]["schema"]]
        text = HEADER.format(n=len(g), keys=", ".join(f'"{t["tag"]}": <answer>' for t in g),
                             shape=shape_text(schema))
        text += "".join(f"\n=== TASK {t['tag']} ===\n{t['prompt'].strip()}\n"
                        f"=== END TASK {t['tag']} ===\n" for t in g)
        (bdir / f"{name}.txt").write_text(text)
        index[name] = {"tags": [t["tag"] for t in g], "keys": {t["tag"]: t["key"] for t in g},
                       "schema": g[0]["schema"],
                       "chars": len(text), "round": g[0]["tag"].split("-")[0]}
    (bdir / "index.json").write_text(json.dumps(index, indent=1))
    return [{"name": k, **v} for k, v in index.items()]


def batches(manual_dir) -> list[dict]:
    """All batches of the current round, each with how many of its tasks are answered."""
    adir, bdir = _dirs(manual_dir)
    p = bdir / "index.json"
    if not p.exists():
        return []
    out = []
    for name, b in json.loads(p.read_text()).items():
        done = sum((adir / f"{b['keys'][t]}.json").exists() for t in b["tags"])
        out.append({"name": name, **b, "answered": done, "complete": done == len(b["tags"]),
                    "text": (bdir / f"{name}.txt").read_text()})
    return out


def _parse(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)      # code fences from chat UIs
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object found in the reply")
    return json.loads(t[a:b + 1])


def save_answer(manual_dir, batch_name: str, text: str) -> tuple[list[str], list[str]]:
    """Check a pasted reply and save each valid task answer. Returns (saved, problems)."""
    adir, bdir = _dirs(manual_dir)
    index = json.loads((bdir / "index.json").read_text())
    if batch_name not in index:
        return [], [f"unknown batch {batch_name}"]
    b = index[batch_name]
    schema = SCHEMAS[b["schema"]]
    try:
        reply = _parse(text)
    except (ValueError, json.JSONDecodeError) as e:
        return [], [f"could not read the reply as JSON ({e}); copy the model's whole reply"]
    if not isinstance(reply, dict):
        return [], ["the reply must be one JSON object with a key per task id"]
    if len(b["tags"]) == 1 and b["tags"][0] not in reply:   # model answered without the id
        reply = {b["tags"][0]: reply}
    keyed = {str(k).strip(): v for k, v in reply.items()}
    saved, problems = [], []
    for tag in b["tags"]:
        if tag not in keyed:
            problems.append(f"{tag}: missing from the reply")
            continue
        try:
            obj = schema.model_validate(keyed[tag])
        except ValidationError as e:
            problems.append(f"{tag}: wrong shape ({str(e).splitlines()[0]})")
            continue
        (adir / f"{b['keys'][tag]}.json").write_text(obj.model_dump_json(indent=1))
        saved.append(tag)
    return saved, problems


def round_name(bs: list[dict]) -> str:
    return ROUND_NAMES.get(bs[0]["round"], bs[0]["round"]) if bs else ""


# ----------------------------------------------------------------------------- Colab UI

def colab_ui(zip_path: str, out: str = "outputs", manual_dir: str = "/content/manual_work",
             label: str = "ChatGPT (web)", hooks: int = 3, episodes: str = "",
             max_chars: int = 40000):
    """One cell, all rounds: shows each batch, takes the pasted reply, checks it, and
    prepares the next round when a round is done."""
    import ipywidgets as w
    from IPython.display import Markdown, display

    cmd = [sys.executable, "run.py", zip_path, "--provider", "manual", "--model", label,
           "--manual-dir", manual_dir, "--hooks", str(hooks), "--max-chars", str(max_chars),
           "--out", out] + (["--episodes", episodes] if episodes else [])
    head = w.HTML()
    prompt = w.Textarea(layout=w.Layout(width="100%", height="180px"))
    reply = w.Textarea(placeholder="Paste the model's whole reply here",
                       layout=w.Layout(width="100%", height="140px"))
    save, skip = w.Button(description="Save reply", button_style="primary"), w.Button(description="Skip for now")
    msg = w.Output()
    state = {"cur": None, "tried": set()}   # tried = batches saved at least partly this round

    def prepare() -> bool:
        """Run the pipeline: it reuses every saved answer and writes the next round."""
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            with msg:
                print(r.stdout[-1500:], r.stderr[-1500:])
            return False
        state["cur"], state["tried"] = None, set()
        return True

    def open_batches():
        return [b for b in batches(manual_dir) if not b["complete"] and b["name"] not in state["tried"]]

    def show():
        allb, todo = batches(manual_dir), open_batches()
        if not allb:                            # nothing pending: final results are written
            head.value = "<h3>All rounds done</h3>"
            prompt.layout.display = reply.layout.display = "none"
            save.layout.display = skip.layout.display = "none"
            msg.clear_output()
            with msg:
                display(Markdown(Path(out, "shortlist.md").read_text()))
            return
        b = next((x for x in todo if x["name"] == state["cur"]), todo[0])
        state["cur"] = b["name"]
        done = sum(x["complete"] or x["name"] in state["tried"] for x in allb)
        head.value = (f"<h3>{round_name([b])}</h3><p><b>Batch {allb.index(b) + 1} of "
                      f"{len(allb)}</b> · {done} done · {len(b['tags'])} tasks, {b['chars']:,} "
                      "characters. Click in the top box, press Ctrl+A then Ctrl+C, paste it into "
                      f"a new chat with {label}, then paste the whole reply in the lower box.</p>")
        prompt.value, reply.value = b["text"], ""

    def next_batch():
        todo = open_batches()
        if not todo:                            # every batch of this round handled
            with msg:
                print("Round done. Preparing the next round...")
            if not prepare():
                return
        else:
            names = [x["name"] for x in batches(manual_dir)]
            after = [x for x in todo if names.index(x["name"]) > names.index(state["cur"])]
            state["cur"] = (after or todo)[0]["name"]
        show()

    def on_save(_):
        b = next(x for x in batches(manual_dir) if x["name"] == state["cur"])
        saved, problems = save_answer(manual_dir, b["name"], reply.value)
        msg.clear_output()
        with msg:
            print(f"{b['name']}: saved {len(saved)} of {len(b['tags'])} answers.")
            for p in problems:
                print("  -", p)
        if not saved:                           # nothing usable: stay on this batch
            with msg:
                print("Nothing saved. Copy the model's whole reply and save again.")
            return
        if problems:
            with msg:
                print("The missing tasks will be asked again in the next round.")
        state["tried"].add(b["name"])
        next_batch()

    def on_skip(_):
        todo = open_batches()
        names = [x["name"] for x in todo]
        if len(todo) > 1:
            state["cur"] = names[(names.index(state["cur"]) + 1) % len(names)]
        show()

    save.on_click(on_save)
    skip.on_click(on_skip)
    display(w.VBox([head, w.HTML("<b>1. Copy this prompt</b>"), prompt,
                    w.HTML("<b>2. Paste the reply</b>"), reply, w.HBox([save, skip]), msg]))
    if prepare():
        show()


# ----------------------------------------------------------------------------- CLI

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "answer", "show"])
    ap.add_argument("batch", nargs="?")
    ap.add_argument("reply_file", nargs="?")
    ap.add_argument("--manual-dir", default="manual")
    a = ap.parse_args()
    if a.action == "status":
        bs = batches(a.manual_dir)
        print(round_name(bs) or "No batches yet: run  python run.py <zip> --provider manual")
        for b in bs:
            print(f"  {b['name']}: {b['answered']}/{len(b['tags'])} answered, {b['chars']:,} chars")
    elif a.action == "show":
        print(next(b["text"] for b in batches(a.manual_dir) if b["name"] == a.batch))
    else:
        saved, problems = save_answer(a.manual_dir, a.batch, Path(a.reply_file).read_text())
        print(f"Saved {len(saved)} answers.")
        for p in problems:
            print("  -", p)
