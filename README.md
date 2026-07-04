# IT Documentation Translator Agent

A multi-agent pipeline built with **Google ADK 2.0** that translates English IT documentation into professional Russian. The system automatically builds and maintains a domain-specific glossary, preserves code blocks, and produces an interactive HTML output with hover tooltips explaining technical terms.

---

## The Problem

Russian-speaking developers and technical teams frequently work with English documentation — API references, deployment guides, architecture descriptions. Machine translation tools produce literal, often awkward results: they mistranslate domain terms ("endpoint" → "конечная точка" instead of "эндпоинт"), destroy code block formatting, and produce text that no native speaker would write.

This agent solves that by combining deterministic preprocessing, automatic glossary management, multiple specialized LLM passes, and interactive output — all orchestrated as a linear ADK 2.0 workflow.

---

## Architecture

```
 User Input (text or screenshot)
          │
          ▼
   ┌─────────────┐
   │   parser    │  Regex: protect ```code```, `inline`, $VAR → placeholders
   │             │  OCR: Gemini Vision extracts text from uploaded screenshots
   └──────┬──────┘
          │
          ▼
   ┌──────────────────┐
   │ glossary_lookup  │  Matches known IT terms from glossary.json (~80 terms)
   └──────┬───────────┘
          │
          ▼
   ┌─────────────────────┐
   │ auto_glossary_fill  │  Gemini: finds new compound IT terms in the text,
   │                     │  translates them, caches to glossary.json
   └──────┬──────────────┘
          │
          ▼
   ┌──────────────────────────┐
   │ prepare_translator_input │  Builds prompt: top-12 glossary terms + text
   └──────┬───────────────────┘
          │
          ▼
   ┌──────────────┐
   │  translator  │  LlmAgent — EN→RU, preserves placeholders and structure
   └──────┬───────┘
          │
          ▼
   ┌────────────────────┐
   │ grammar_corrector  │  LlmAgent — fixes case/declension, removes calques
   │                    │  e.g. "гранулярность" → "детализация"
   └──────┬─────────────┘
          │
          ▼
   ┌──────────────────────────┐
   │ prepare_reviewer_input   │  Builds QA prompt with source + translation
   └──────┬───────────────────┘
          │
          ▼
   ┌──────────┐
   │ reviewer │  LlmAgent — verifies placeholders, glossary consistency,
   │          │  flags genuinely ambiguous terms
   └──────┬───┘
          │
          ▼
   ┌──────────────┐
   │ reassembler  │  Restores code blocks; fuzzy fallback for LLM index drift
   └──────┬───────┘
          │
          ▼
   ┌───────────────────┐
   │ tooltip_generator │  Single batch Gemini call → 1-sentence Russian
   │                   │  definitions for all matched glossary terms
   └──────┬────────────┘
          │
          ▼
   ┌────────────────┐
   │ html_formatter │  Converts to styled HTML; wraps terms in CSS hover
   │                │  tooltips; saves output.html
   └────────────────┘
```

**10 nodes total:** 3 LlmAgents + 7 deterministic/direct-Gemini nodes, wired as a linear ADK `Workflow`.

---

## Key Features

- **Code block protection** — triple-backtick blocks, inline code, and variable names are replaced with typed placeholders (`[CODE_BLOCK_N]`, `[CLI_COMMAND_N]`, `[VAR_NAME_N]`) before translation and restored exactly after. A fuzzy fallback handles cases where the LLM shifts a placeholder index.
- **Auto-growing glossary** — on each translation, a Gemini call extracts new IT compound terms from the text, translates them, and caches them in `glossary.json`. The glossary starts at ~26 terms and grows with usage.
- **OCR support** — if a screenshot is uploaded in the ADK Playground, Gemini Vision extracts the text before the translation pipeline runs.
- **Grammar correction pass** — a dedicated LlmAgent fixes Russian case/declension errors and replaces unnatural direct transliterations ("имплементация" → "реализация").
- **Interactive HTML output** — `output.html` renders the translation with syntax-highlighted code blocks and CSS hover tooltips on technical terms. Tooltips are generated in a single batch Gemini call, with declined-form matching (e.g. "биллинга" matches the stem "биллинг").

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | Google ADK 2.0 (`google-adk`) |
| LLM | Gemini 2.0 Flash (`gemini-2.0-flash`) |
| Multi-agent orchestration | ADK `Workflow` with `@node` decorator |
| Direct LLM calls | `google.genai.Client` (outside ADK for glossary ops) |
| OCR | Gemini Vision (`inline_data` / `types.Blob`) |
| Schema validation | Pydantic v2 |
| Package management | `uv` |
| Local dev server | `uv run adk web .` |
| Containerisation | Docker (`Dockerfile` included) |

---

## Setup

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 1. Clone and install

```bash
git clone https://github.com/MCR34/agentic-translator.git
cd agentic-translator/translation-agent
uv sync
```

### 2. Configure environment

Create `.env` in `translation-agent/` (never commit this file):

```env
GEMINI_API_KEY=your_key_here
GOOGLE_GENAI_USE_ENTERPRISE=FALSE
```

### 3. Run

```bash
uv run adk web .
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### 4. Translate

Type your text in the chat (optionally prefix with `Translate:`):

```
Translate: The CI/CD pipeline triggers on every push to main. It builds a Docker
image tagged with the commit SHA and pushes it to the registry. The `DEPLOY_ENV`
variable must be set to `production` before a rollout restart.
```

After the pipeline completes, open `translation-agent/output.html` in a browser to see the styled translation with hover tooltips.

You can also upload a screenshot — the agent will OCR it with Gemini Vision.

---

## Docker

```bash
cd translation-agent
docker build -t it-translator .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key it-translator
```

---

## Project Structure

```
agentic-translator/
├── README.md
├── KAGGLE_WRITEUP.md
└── translation-agent/
    ├── app/
    │   ├── agent.py          # All 10 workflow nodes + LlmAgents + helpers
    │   ├── config.py         # MODEL_NAME, GLOSSARY_PATH constants
    │   └── app_utils/        # Telemetry and typing helpers
    ├── glossary.json         # Auto-growing IT term dictionary (EN→RU)
    ├── Dockerfile
    ├── pyproject.toml
    └── tests/
```

---

## ADK Concepts Demonstrated

| Concept | Where |
|---|---|
| Multi-agent system (ADK Workflow) | `app/agent.py` — 10-node linear graph |
| LlmAgent with output schema | `translator`, `grammar_corrector`, `reviewer` |
| Direct Gemini calls (non-ADK) | `_gemini_call()`, `_extract_text_from_image()` |
| Session state passing | `ctx.state` / `EventActions(state_delta=...)` |
| Gemini Vision (OCR) | `_extract_text_from_image()` in `parser` node |
| Security: no secrets in code | All keys via `.env` / environment variables |
| Deployability | `Dockerfile` + `agents-cli-manifest.yaml` |
| Agent CLI | `agents-cli playground` / `uv run adk web .` |
