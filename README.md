# Promo Extractor — Kuku AI Generalist Assignment, Problem 1

Repository: https://github.com/ShubNoob/promo-extractor · Author: Shubham Agarwal (230996)

Reads a microdrama's screenplays and returns ranked 20–60 second promo clips that open
on a hook (first 3 seconds), cut on a cliffhanger, and make sense to someone who has never
seen the show. For every promo: episode, start/end timestamps, hook, cliffhanger, one line
on why it works, a tier and a posting order.

## One pipeline, three ways to run the judging

Python owns every timestamp in every mode: each time the judge returns is checked against
the script, and the 20–60 s and 3-second rules are enforced in code. Only steps 2, 4 and 6
change.

| Step | What it does | Who does it |
|---|---|---|
| 1 Parse | script → timeline of camera / action / dialogue / sound events, times in seconds, shot cuts | Python |
| 2 Find hooks | 3–5 hook moments per episode, the line that causes the moment | LLM (1 task per episode) |
| 3 Frame | hook → window up to 60 s; rejects times that match no line; hook within 3 s | Python |
| 4 Judge | pick the cut, score hook / cliffhanger / middle, "never seen the show" test, write hook, cliffhanger, why | LLM (1 task per clip) |
| 5 Check the cut | snap the end to a real line end or shot cut inside 20–60 s; log every fix | Python |
| 6 Sort and rank | Tier 1 ready / Tier 2 trim / Tier 3 rejected with reason; posting order | Python tiers + LLM ranking (1 task) |
| 7 Hand check | human labels the top picks blind; `--score` reports agreement | Human |

| Mode | `--provider` | LLM? | Key? | Used for |
|---|---|---|---|---|
| **Manual** (submitted run) | `manual` | yes: any chat model in the browser | no | the tool writes the LLM tasks into a few large prompts; you paste each into ChatGPT / Claude / Gemini and paste the reply back; Python checks every answer |
| API | `openai`, `anthropic`, `gemini` | yes, called directly | yes | the same tasks with no copy-paste: built and tested, not run on the pack (no API credits; Gemini free tier overloaded) |
| Rules | `rules` | no | no | a no-AI baseline from script signals (emotion tags, action verbs, `!`, `?`, `...`); powers the web app |

## Run it

**Colab (manual mode):** open `notebook.ipynb` in Colab (File → Open notebook → GitHub →
`ShubNoob/promo-extractor`), run the cells in order, upload the scripts
zip when asked. Section 5 shows each prompt with a box to paste the reply into, prepares the
next round by itself (find hooks → judge clips → rank) and ends on the shortlist. With 3
hooks per episode, a 50-episode show is about 30–35 pastes at 40,000 characters each, or about
15 with Claude.ai at 90,000.

**Manual mode from the command line:**
```bash
python run.py scripts.zip --provider manual --model "ChatGPT (web)" --hooks 3   # prepares a round
python manual.py status                                    # batches answered / pending
python manual.py show batch_01 > prompt.txt                # paste this into the chat model
python manual.py answer batch_01 reply.txt                 # save the model's reply
python run.py scripts.zip --provider manual --model "ChatGPT (web)" --hooks 3   # next round / final results
```

**Other commands:**
```bash
pip install -r requirements.txt
python test_checks.py                             # 17 checks
python run.py scripts.zip --provider rules        # no-AI baseline, seconds
export OPENAI_API_KEY=...                         # API mode
python run.py scripts.zip --provider openai --model gpt-5-mini --fallback gpt-4o-mini
python run.py --score outputs/hand_check.csv      # after labelling the hand-check sheet
streamlit run app.py                              # web app
```
`--provider anthropic` (key `ANTHROPIC_API_KEY`) and `--provider gemini` (`GEMINI_API_KEY`)
work like OpenAI. Defaults live in `config.yaml`.

**Web app:** live at [FILL: https://<your-app>.streamlit.app]. Open a `promos.json` (and your
labelled hand check) to browse the ranked posting queue, or upload scripts and run them in the
browser: copy-paste with ChatGPT or Claude (LLM, no key), instant rules, or an API key. To host
your own copy: on [share.streamlit.io](https://share.streamlit.io) create an app from this repo
with `app.py` as the main file.

## Outputs (`outputs/`)

| File | What |
|---|---|
| `shortlist.md` | Tier 1 in posting order, then Tier 2 and Tier 3 with reasons |
| `promos.json` | every judged clip with segments, scores, notes and run metadata |
| `promos.csv` | the same, one row per clip |
| `failures.csv` | hooks rejected by the checks, Tier 3 reasons, corrected timestamps |
| `hand_check.csv` | top picks without scores, for blind human labelling |
| `run_log.json` | counts, timings, model calls and tokens (API mode), tasks still pending (manual mode), config used |

## Scale and cost

One show is about 200–300 LLM tasks (50 to find hooks, 150–250 to judge, 1 to rank).
Manual mode packs them into a few dozen pastes and costs nothing; API mode runs them
unattended, and `run_log.json` records the tokens, so cost per show = tokens × the model's
price. Answers are saved, so an interrupted run resumes without repeating work; in API mode a
backup model takes over when the main one is overloaded or rate-limited, and the run stops
cleanly when credits or quota run out. Rules mode runs a show in seconds at zero cost.

## Files

`run.py` pipeline · `parser.py` scripts → timeline · `framer.py` windows + timestamp checks ·
`llm_nodes.py` model wrapper, prompts and the rules engine · `manual.py` manual mode (batches, paste-back checks, Colab UI) · `ranker.py` tiers, ranking,
outputs · `app.py` Streamlit dashboard · `test_checks.py` tests · `config.yaml` settings ·
`rule_maker.py` + `reference_promos.txt` optional, not used in the submitted run: learns hook
and cliffhanger patterns from 60 ReelShort / DramaBox promo Shorts with a video-capable model.

## Known limits

- Works from the script, not the video: it can't judge acting, pacing on screen or audio.
- One continuous segment per promo, inside one episode.
- The LLM grades its own picks (hooks and cuts come from the same model), so the blind
  hand check matters.
- Manual mode depends on the person pasting the whole reply; replies are checked against the
  answer format, and missing or malformed tasks are asked again.
- Rules mode misses quiet hooks with no emotion tag or keyword (e.g. "Because we know who
  he is."), and its "never seen the show" test only catches unseen names.

## Confidentiality

Kuku's scripts are confidential: `.gitignore` keeps scripts, outputs, the cache and manual
answers out of the repo. Upload the zip at run time. Manual mode sends script text to the
chat model you paste into, as API mode would.
