# IT Documentation Translator: A Multi-Agent Pipeline for EN→RU Technical Localization

**Subtitle:** An ADK 2.0 workflow that automatically builds a domain glossary, preserves code formatting, and produces interactive HTML with hover-tooltip term explanations

**Track:** Agents for Business

---

## The Problem

Russian-speaking engineering teams work with English technical documentation every day — API references, deployment guides, architecture decisions, runbooks. Existing machine translation tools produce text that feels wrong to a native speaker: they mistranslate domain-specific terms ("endpoint" becomes "конечная точка" instead of the industry-standard "эндпоинт"), break code block formatting, and generate grammatically awkward sentences that no professional would write.

The core challenge is not raw translation quality — modern LLMs can produce fluent text. The challenge is **domain consistency**: a single document may contain dozens of specialized IT terms, each of which has an accepted Russian equivalent in the industry. Without a managed glossary, different sentences translate the same term differently, which is unprofessional and confusing.

A secondary challenge is **structural preservation**: technical documentation contains code blocks, inline variable names, CLI commands, and environment variables. A translator that modifies these destroys the document's usability.

This project addresses both challenges through a pipeline of specialized agents, each responsible for one well-defined task.

---

## Why Agents?

A single LLM prompt cannot reliably solve this problem. The translation task requires:

1. **Deterministic preprocessing** — code extraction must be exact; a regex is more reliable than an LLM for this.
2. **Glossary management** — detecting new domain terms and caching their translations requires a separate reasoning step before translation.
3. **Post-processing validation** — checking that code blocks survived the translation pass requires comparing source and target, which is its own cognitive task.
4. **Output formatting** — generating interactive HTML with term explanations is unrelated to translation itself.

Each of these is cleanly separable. Agents allow us to assign the right tool to each task: LlmAgents where language understanding is needed, direct Gemini calls where a lightweight structured response is enough, and pure Python nodes where determinism is required. The ADK Workflow orchestrates them as a linear pipeline with shared session state.

---

## Architecture

The pipeline consists of **10 nodes** wired as a linear ADK `Workflow`:

### Node 1 — `parser` (Python `@node`)
Extracts raw text from the ADK input (handling `str`, `Content`, `Part`, and `list` types). Uses regex to find and replace:
- Triple-backtick code blocks → `[CODE_BLOCK_N]`
- Inline code containing CLI commands → `[CLI_COMMAND_N]`
- Other inline code (variable names, flags) → `[VAR_NAME_N]`

If the input contains an image (uploaded screenshot), the node calls Gemini Vision via `google.genai.Client` to OCR the image before processing. Reconstructing a clean `types.Part(inline_data=types.Blob(...))` is required to strip the `display_name` field that ADK Playground adds but the Developer API rejects.

The protected blocks and their original content are saved to session state.

### Node 2 — `glossary_lookup` (Python `@node`)
Reads `glossary.json` and performs word-boundary regex matching against the parsed text. Returns a list of `GlossaryMatch` objects for all found terms. The glossary starts with ~26 hand-curated terms and grows automatically.

### Node 3 — `auto_glossary_fill` (Python `@node` + direct Gemini)
Makes two types of Gemini calls before translation runs:
- **One extraction call**: asks Gemini to find new compound IT terms in the document that are not yet in the glossary. A strict prompt filters out common English words, proper nouns, and placeholder tags. Capped at 5 new terms per request.
- **One translation call per term**: gets the Russian equivalent, rejects translations longer than 5 words (indicating confusion). Saves accepted terms to `glossary.json` and extends the match list for the current translation.

### Node 4 — `prepare_translator_input` (Python `@node`)
Builds the translator's prompt by selecting the 12 most relevant glossary terms (sorted by English term length, so compound terms like "CI/CD pipeline" take priority over single words) and prepending them as a glossary block.

### Node 5 — `translator` (LlmAgent)
Translates the text with two critical constraints injected via the system instruction: placeholders must be copied character-for-character including the exact index suffix, and paragraph/line structure must be preserved identically.

### Node 6 — `grammar_corrector` (LlmAgent)
A native-speaker editing pass that fixes grammatical errors (case, declension, gender agreement) and replaces unnatural direct transliterations — for example, "гранулярность" becomes "детализация".

### Node 7 — `prepare_reviewer_input` (Python `@node`)
Composes the review prompt with both the English source and Russian translation for side-by-side comparison.

### Node 8 — `reviewer` (LlmAgent)
Quality-checks the translation: verifies placeholder integrity, checks glossary consistency, and flags only genuinely ambiguous IT terms (those with multiple competing accepted Russian translations in active professional use). The instruction explicitly prohibits flagging single-accepted transliterations like webhook→вебхук.

### Node 9 — `reassembler` (Python `@node`)
Restores the original code block content. Uses exact matching first; if an LLM shifted a placeholder index (e.g., `[CODE_BLOCK_0]` became `[CODE_BLOCK_1]`), a fuzzy regex fallback matches by block type and substitutes by position.

### Node 10 — `tooltip_generator` + `html_formatter` (Python `@node` + direct Gemini)
A single batch Gemini call generates one-sentence Russian definitions for all matched glossary terms. The `html_formatter` then converts the translation to styled HTML: triple-backtick blocks → `<pre><code>`, inline code → `<code>`, and recognized terms are wrapped in `<span class="tt" data-tip="...">` for CSS hover tooltips. Term matching handles Russian morphology: for single-word terms, the pattern matches the stem plus up to 2-character noun endings (e.g. "биллинг" matches "биллинга"), blocking accidental matches on adjective forms. The output is saved as `output.html`.

---

## Key Technical Decisions

**Placeholder approach over prompt-only protection.** Telling the LLM "don't translate code blocks" works most of the time but not always. Converting code to typed placeholders before the LLM sees them makes the protection deterministic. The translator never sees the code content — it sees `[CODE_BLOCK_0]` and passes it through by design.

**Glossary-first, then translate.** Running `auto_glossary_fill` before translation (not after) ensures the translator has consistent term mappings from the first pass. This eliminates a common failure mode where the same term is translated differently in different sentences of the same document.

**Capped glossary injection.** Early testing showed that passing all 75+ glossary terms to the translator caused it to "lock up" — trying to apply so many constraints produced garbled output. Capping at the 12 longest (most specific) terms resolved this.

**Fuzzy placeholder recovery.** LLMs occasionally shift numeric indices: they receive `[CODE_BLOCK_0]` and output `[CODE_BLOCK_1]`. The reassembler's second pass matches by block type when exact index matching fails, preventing silent placeholder loss.

**Pydantic `additionalProperties` stripping.** The Gemini Developer API rejects JSON schemas that include `additionalProperties`. All output schemas inherit from a custom `ADKModel` base class that recursively strips this field from the schema before ADK sends it to the API.

---

## Course Concepts Applied

| Concept | Implementation |
|---|---|
| **Multi-agent system (ADK)** | 10-node `Workflow`; 3 `LlmAgent` instances + 7 Python/Gemini nodes |
| **Security** | All API keys via `.env` / environment variables; `.gitignore` excludes secrets |
| **Deployability** | `Dockerfile` included; `agents-cli-manifest.yaml` for Agent CLI deployment |
| **Agent skills / CLI** | `agents-cli playground` / `uv run adk web .`; manifest configured |

---

## Results

The system successfully translates complex IT documentation including multi-paragraph texts with nested code blocks, numbered lists, inline environment variable references, and Kubernetes manifests. Key outcomes:

- Code blocks (bash, JSON, YAML) are preserved exactly and rendered with syntax highlighting in the HTML output.
- Domain terms are translated consistently throughout a document: "endpoint" always becomes "эндпоинт", "load balancer" always becomes "балансировщик нагрузки".
- The interactive HTML viewer shows hover tooltips on technical terms, making the translation useful for readers who encounter unfamiliar terminology.
- The glossary grows automatically: started at 26 terms, reached 80+ after typical usage without manual curation.

---

## Challenges and Lessons Learned

**The HITL trap.** The initial design included a human-in-the-loop step where the user could correct uncertain term translations. In practice, the ADK Playground interface made it confusing — users typed in the wrong input field. We removed HITL entirely in favour of automatic glossary enrichment, which proved more reliable.

**Glossary quality over quantity.** Early versions added up to 10 new terms per request. This caused the translator to receive too many constraints and produced garbled output. Reducing the cap to 5 and adding strict extraction prompts (no common words, no product names, compound IT terms only) fixed the issue.

**Russian morphology.** Matching translated terms for tooltip insertion is complicated by Russian declension — "биллинг" appears as "биллинга" in genitive case. A simple `str.replace()` missed declined forms entirely. The solution uses a regex stem-plus-suffix approach with a length cap on suffixes to distinguish noun declension from adjective forms.

---

## What's Next

- Extend to Spanish and Chinese via a language configuration parameter.
- Add a web viewer frontend that calls the ADK HTTP API directly, eliminating the copy-to-browser step for `output.html`.
- Cloud deployment via `agents-cli deploy` to make the service publicly accessible without local setup.
