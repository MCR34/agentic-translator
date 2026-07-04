# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo
import os
import json
import urllib.request
import urllib.parse
import re
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

load_dotenv()

import google.auth
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, START, node
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

# Load project config
from .config import MODEL_NAME, GLOSSARY_PATH

# Set GCP environments
try:
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "mock-project-id")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
else:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


# --- Schema Definitions ---

def _strip_additional_properties(obj: Any) -> None:
    """Recursively remove additionalProperties from a JSON schema dict.
    Required because Gemini Developer API (non-Vertex AI) rejects this field."""
    if isinstance(obj, dict):
        obj.pop('additionalProperties', None)
        for v in obj.values():
            _strip_additional_properties(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_additional_properties(item)


class ADKModel(BaseModel):
    """Base model that strips additionalProperties from JSON schema for Gemini Developer API compatibility."""
    model_config = ConfigDict(extra='ignore')

    @classmethod
    def model_json_schema(cls, **kwargs) -> dict:
        schema = super().model_json_schema(**kwargs)
        _strip_additional_properties(schema)
        return schema


class ParserOutput(ADKModel):
    document_text: str = Field(
        description="The parsed document text with unique placeholders in the format [CODE_BLOCK_N], [CLI_COMMAND_N], or [VAR_NAME_N]."
    )
    protected_blocks: Dict[str, str] = Field(
        description="Mapping of placeholders (e.g. '[CODE_BLOCK_0]') to their original content."
    )


class GlossaryMatch(ADKModel):
    english: str
    russian: str


class GlossaryLookupOutput(ADKModel):
    matching_terms: List[GlossaryMatch]


class PrepareTranslatorInputOutput(ADKModel):
    prompt: str


class TranslationOutput(ADKModel):
    translated_text: str


class GrammarCorrectorOutput(ADKModel):
    translated_text: str = Field(
        description="The Russian translation after grammar correction — case agreements, declension, and verb forms fixed. Placeholders like [CLI_COMMAND_N] must remain untouched."
    )


class PrepareReviewerInputOutput(ADKModel):
    prompt: str


class ReviewerOutput(ADKModel):
    is_valid: bool = Field(
        description="True if the translation is consistent, code blocks are unaltered, and there are no ambiguous terms needing human review."
    )
    feedback: str = Field(
        description="Detailed feedback describing any errors, inconsistencies, or flagged terms."
    )
    ambiguous_terms: List[str] = Field(
        default_factory=list,
        description="A list of English terms from the text that are ambiguous, not in the glossary, or need human review.",
    )


# --- Node Implementations ---

_CLI_TOOLS = re.compile(
    r'^(?:npm|yarn|pip|pip3|git|docker|kubectl|bash|sh|zsh|python|python3|'
    r'node|make|cargo|go|mvn|gradle|apt|apt-get|brew|curl|wget|ssh|scp|'
    r'cd|ls|cp|mv|rm|mkdir|cat|echo|export|source|chmod|chown)\b'
)


def _gemini_call(prompt: str) -> str:
    """Lightweight direct Gemini call for term extraction and translation (no ADK overhead)."""
    from google import genai
    client = genai.Client()
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return resp.text.strip()


def _extract_text_from_image(image_part: Any) -> str:
    """Use Gemini Vision to OCR a screenshot and return the extracted text."""
    from google import genai
    client = genai.Client()
    # Reconstruct a clean Part — ADK Playground adds display_name which the
    # Developer API rejects. Strip it by creating a new Part from raw bytes only.
    clean_part = types.Part(
        inline_data=types.Blob(
            mime_type=image_part.inline_data.mime_type,
            data=image_part.inline_data.data,
        )
    )
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            clean_part,
            "Extract all text from this screenshot exactly as it appears. "
            "Preserve code blocks, command lines, and formatting. "
            "Return only the raw text content, no commentary.",
        ],
    )
    return resp.text.strip()


_OUTPUT_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output.html')

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Технический перевод</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:860px;margin:40px auto;padding:0 24px;color:#1a1a2e;line-height:1.75;background:#f0f4f8}}
.card{{background:#fff;border-radius:14px;padding:40px 52px;box-shadow:0 2px 20px rgba(0,0,0,.09)}}
.header{{display:flex;align-items:center;gap:10px;margin-bottom:28px;padding-bottom:16px;border-bottom:1px solid #e9ecef}}
.header h1{{margin:0;font-size:1em;font-weight:600;color:#495057}}
.badge{{background:#e8f4fd;color:#0969da;font-size:.72em;padding:3px 10px;border-radius:12px;font-weight:500}}
p{{margin:0 0 1.1em}}
pre{{background:#1e1e2e;color:#cdd6f4;padding:18px 22px;border-radius:10px;overflow-x:auto;font-family:'Consolas','Fira Code',monospace;font-size:.875em;line-height:1.55;margin:1.3em 0}}
code{{background:#e8f4fd;color:#0969da;padding:2px 7px;border-radius:4px;font-family:'Consolas','Fira Code',monospace;font-size:.9em}}
pre code{{background:none;color:inherit;padding:0;font-size:1em}}
.tt{{border-bottom:1.5px dashed #0969da;color:#0969da;cursor:help;position:relative;text-decoration:none}}
.tt::after{{content:attr(data-tip);position:absolute;bottom:130%;left:50%;transform:translateX(-50%);background:#1e1e2e;color:#e2e8f0;padding:9px 13px;border-radius:7px;font-size:.8em;width:290px;white-space:normal;line-height:1.45;pointer-events:none;opacity:0;transition:opacity .18s ease;z-index:1000;box-shadow:0 6px 16px rgba(0,0,0,.35);text-align:left}}
.tt::before{{content:'';position:absolute;bottom:120%;left:50%;transform:translateX(-50%);border:5px solid transparent;border-top-color:#1e1e2e;opacity:0;transition:opacity .18s ease;pointer-events:none}}
.tt:hover::after,.tt:hover::before{{opacity:1}}
</style>
</head>
<body>
<div class="card">
<div class="header"><h1>Технический перевод</h1><span class="badge">EN → RU</span></div>
{content}
</div>
</body>
</html>"""


def _text_to_html(text: str, tooltip_data: Dict[str, str]) -> str:
    from html import escape
    result = []
    segments = re.split(r'(```\w*\n?[\s\S]*?```)', text)
    for seg in segments:
        cb = re.match(r'```(\w*)\n?([\s\S]*?)```', seg)
        if cb:
            lang = cb.group(1) or "text"
            code = escape(cb.group(2).rstrip())
            result.append(f'<pre><code class="lang-{lang}">{code}</code></pre>')
        else:
            # Save inline codes with null-byte sentinels before any processing
            inlines: Dict[str, str] = {}
            ic_n = [0]

            def save_ic(m: re.Match, _n=ic_n, _d=inlines) -> str:
                key = f"\x00IC{_n[0]}\x00"
                _d[key] = f'<code>{escape(m.group(1))}</code>'
                _n[0] += 1
                return key

            seg = re.sub(r'`([^`\n]+)`', save_ic, seg)

            # Wrap tooltip terms IN RAW TEXT before HTML escaping so that:
            # a) declined forms are captured ("биллинга" matches stem "биллинг")
            # b) sentinels survive html.escape() unchanged (they're control chars)
            # Suffix length limit: single-word terms allow ≤2 suffix chars (noun declension: -а, -у,
            # -е, -ом) but block adjective endings (-ной, -ного, -ным = 3+ chars). Multi-word
            # compound terms allow any suffix on the first word since the rest anchors the match.
            for rus, tip in sorted(tooltip_data.items(), key=lambda x: -len(x[0])):
                is_compound = ' ' in rus
                suffix = r'[а-яёА-ЯЁ]*' if is_compound else r'[а-яёА-ЯЁ]{0,2}'
                pattern = re.compile(
                    r'(?<![а-яёА-ЯЁa-zA-Z\-])' + re.escape(rus) + suffix,
                    re.UNICODE,
                )
                def make_sentinel(m: re.Match, _tip=tip) -> str:
                    return f'\x01TT\x02{_tip}\x03{m.group(0)}\x04'
                seg = pattern.sub(make_sentinel, seg, count=1)

            # Escape HTML — also escapes tip text inside sentinels (correct for attributes)
            seg = escape(seg)

            # Restore inline codes
            for key, html in inlines.items():
                seg = seg.replace(key, html)

            # Restore tooltip sentinels → <span> tags
            def restore_tt(m: re.Match) -> str:
                return f'<span class="tt" data-tip="{m.group(1)}">{m.group(2)}</span>'
            seg = re.sub('\x01TT\x02([^\x03]*)\x03([^\x04]*)\x04', restore_tt, seg)

            for para in re.split(r'\n{2,}', seg):
                para = para.strip()
                if para:
                    result.append(f'<p>{para.replace(chr(10), "<br>")}</p>')

    return '\n'.join(result)


# 1. parser — regex-based (deterministic) extraction of code blocks, CLI commands, variable names.
@node
def parser(ctx: Context, node_input: Any) -> Event:
    # Extract plain text from whatever ADK passes (str, Content, Part, list, etc.)
    if isinstance(node_input, str):
        raw = node_input
    elif hasattr(node_input, 'parts'):
        raw = '\n'.join(
            p.text for p in node_input.parts
            if hasattr(p, 'text') and p.text is not None
        )
    elif hasattr(node_input, 'text'):
        raw = node_input.text
    elif isinstance(node_input, list):
        pieces = []
        for item in node_input:
            if hasattr(item, 'parts'):
                pieces.extend(
                    p.text for p in item.parts
                    if hasattr(p, 'text') and p.text is not None
                )
            elif hasattr(item, 'text'):
                pieces.append(item.text)
            else:
                pieces.append(str(item))
        raw = '\n'.join(p for p in pieces if p)
    else:
        raw = str(node_input)

    # If an image was uploaded, OCR it with Gemini Vision and use that as the source text
    if not raw:
        image_part = None
        if hasattr(node_input, 'parts'):
            image_part = next(
                (p for p in node_input.parts
                 if hasattr(p, 'inline_data') and p.inline_data),
                None,
            )
        elif isinstance(node_input, list):
            for item in node_input:
                if hasattr(item, 'parts'):
                    image_part = next(
                        (p for p in item.parts
                         if hasattr(p, 'inline_data') and p.inline_data),
                        None,
                    )
                    if image_part:
                        break
        if image_part:
            raw = _extract_text_from_image(image_part)

    raw = raw.strip()
    text = re.sub(r'^Translate:\s*', '', raw, flags=re.IGNORECASE).strip()

    document_text = text
    protected_blocks: Dict[str, str] = {}
    code_n = cli_n = var_n = 0

    def replace_triple(m: re.Match) -> str:
        nonlocal code_n
        key = f"[CODE_BLOCK_{code_n}]"
        protected_blocks[key] = m.group(0)
        code_n += 1
        return key

    document_text = re.sub(r'```[\s\S]*?```', replace_triple, document_text)

    def replace_inline(m: re.Match) -> str:
        nonlocal cli_n, var_n
        content = m.group(1)
        if _CLI_TOOLS.match(content.strip()):
            key = f"[CLI_COMMAND_{cli_n}]"
            protected_blocks[key] = m.group(0)
            cli_n += 1
        else:
            key = f"[VAR_NAME_{var_n}]"
            protected_blocks[key] = m.group(0)
            var_n += 1
        return key

    document_text = re.sub(r'`([^`\n]+)`', replace_inline, document_text)

    # Detect "naked" CLI command lines — not wrapped in backticks in the source.
    # A naked code line starts with a known CLI tool but is NOT followed by a verb
    # ("is", "are", "was"...), which would indicate prose rather than a command.
    # Consecutive such lines are grouped into a single CODE_BLOCK placeholder.
    _NAKED_CMD = re.compile(
        r'^(?:python3?|pip3?|npm|yarn|git|docker|kubectl|bash|sh|zsh|node|make|uv|'
        r'curl|wget|virtualenv|venv|cargo|go)\b'
        r'(?!\s+(?:is|are|was|were|has|have|can|will|does|do|the\s|a\s)\b)',
        re.IGNORECASE,
    )
    lines = document_text.split('\n')
    result_lines: List[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if _NAKED_CMD.match(stripped):
            block: List[str] = []
            while idx < len(lines) and (
                _NAKED_CMD.match(lines[idx].strip())
                # allow one blank separator inside a command group
                or (not lines[idx].strip()
                    and idx + 1 < len(lines)
                    and _NAKED_CMD.match(lines[idx + 1].strip()))
            ):
                block.append(lines[idx])
                idx += 1
            while block and not block[-1].strip():
                block.pop()
            key = f"[CODE_BLOCK_{code_n}]"
            protected_blocks[key] = "```\n" + "\n".join(block) + "\n```"
            code_n += 1
            result_lines.append(key)
        else:
            result_lines.append(lines[idx])
            idx += 1
    document_text = '\n'.join(result_lines)

    output = ParserOutput(document_text=document_text, protected_blocks=protected_blocks)
    return Event(
        output=output,
        actions=EventActions(state_delta={"parsed_data": output.model_dump()}),
    )


# 2. glossary_lookup — a tool node that reads glossary.json and surfaces matching terms.
@node
def glossary_lookup(ctx: Context, node_input: ParserOutput) -> Event:
    try:
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            glossary = json.load(f)
    except Exception:
        glossary = {}

    matching_terms = []
    text_lower = node_input.document_text.lower()
    for eng, rus in glossary.items():
        if rus == "unresolved":
            continue
        pattern = r"\b" + re.escape(eng.lower()) + r"\b"
        if re.search(pattern, text_lower):
            matching_terms.append(GlossaryMatch(english=eng, russian=rus))

    return Event(
        output=GlossaryLookupOutput(matching_terms=matching_terms),
        actions=EventActions(
            state_delta={
                "parsed_data": node_input.model_dump(),
                "glossary_terms": [m.model_dump() for m in matching_terms],
                "resolved_terms": {},
            }
        ),
    )


# 2b. auto_glossary_fill — finds new IT terms via Gemini and caches their translations
# in glossary.json before the main translation runs, ensuring consistency from the first pass.
@node
def auto_glossary_fill(ctx: Context, node_input: GlossaryLookupOutput) -> Event:
    parsed_data = ctx.state.get("parsed_data")
    if not parsed_data:
        return Event(output=node_input, actions=EventActions(state_delta={}))

    parsed = ParserOutput(**parsed_data)

    try:
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            glossary = json.load(f)
    except Exception:
        glossary = {}

    known = set(glossary.keys())

    # Step 1: one Gemini call to extract all new IT terms from the document
    extraction_prompt = (
        "List ONLY terms from this text that belong exclusively to software engineering vocabulary "
        "and would appear in a software/IT glossary.\n"
        "Return ONLY a JSON array of English terms, no explanation.\n"
        "Example output: [\"load balancer\", \"microservice\", \"message broker\"]\n"
        f"Skip these already-known terms: {sorted(known)}\n"
        "Skip placeholder tags like [CODE_BLOCK_0], [CLI_COMMAND_0], [VAR_NAME_0].\n"
        "Skip common English words that exist in a general dictionary: "
        "developer, user, project, team, process, feature, release, version, environment, "
        "service, server, client, request, response, data, file, system, application, "
        "test, build (unless part of a compound IT term like 'build pipeline').\n"
        "Skip product/tool proper nouns: Docker, Git, npm, Linux, Windows — they stay as-is.\n\n"
        f"Text:\n{parsed.document_text}"
    )
    new_terms: List[str] = []
    try:
        raw = _gemini_call(extraction_prompt)
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            candidates = json.loads(m.group(0))
            new_terms = [t for t in candidates if isinstance(t, str) and t not in known][:5]
    except Exception:
        pass

    # Step 2: one Gemini call per new term to get the Russian translation
    newly_added: Dict[str, str] = {}
    for term in new_terms:
        try:
            translation = _gemini_call(
                "Translate this IT/software term to Russian for professional technical documentation.\n"
                "Return ONLY the Russian translation (one word or short phrase), nothing else.\n"
                "If the term is typically kept as-is in Russian IT (API, JSON, HTTP, SDK, URL, CLI, "
                "Git, Docker, Linux), return it unchanged.\n"
                f"Term: {term}"
            )
            # Reject suspiciously long or empty translations
            if translation and len(translation.split()) <= 5:
                glossary[term] = translation
                newly_added[term] = translation
        except Exception:
            pass

    if newly_added:
        try:
            with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"auto_glossary_fill: error saving glossary: {e}")

    # Extend matching_terms with the new entries that appear in the text
    all_matching = list(node_input.matching_terms)
    text_lower = parsed.document_text.lower()
    for eng, rus in newly_added.items():
        if re.search(r'\b' + re.escape(eng.lower()) + r'\b', text_lower):
            all_matching.append(GlossaryMatch(english=eng, russian=rus))

    updated = GlossaryLookupOutput(matching_terms=all_matching)
    return Event(
        output=updated,
        actions=EventActions(
            state_delta={"glossary_terms": [m.model_dump() for m in all_matching]},
        ),
    )


# 3. Translator Preparation Node
@node
def prepare_translator_input(
    ctx: Context, node_input: GlossaryLookupOutput
) -> PrepareTranslatorInputOutput:
    parsed_data = ctx.state.get("parsed_data")
    parsed = (
        ParserOutput(**parsed_data)
        if parsed_data
        else ParserOutput(document_text="", protected_blocks={})
    )

    glossary_str = ""
    if node_input.matching_terms:
        # Sort by term length descending (longer/compound terms are more specific and important),
        # cap at 12 to avoid overwhelming the translator with too many constraints.
        top_terms = sorted(node_input.matching_terms, key=lambda t: -len(t.english))[:12]
        glossary_str = "GLOSSARY TERMS (use these translations consistently):\n" + "\n".join(
            f"- {t.english} -> {t.russian}" for t in top_terms
        )

    prompt = f"""{glossary_str}

Text to translate:
{parsed.document_text}
"""
    return PrepareTranslatorInputOutput(prompt=prompt)


# 3. translator — translates the non-code text into Russian.
translator = LlmAgent(
    name="translator",
    model=Gemini(model=MODEL_NAME),
    instruction="""You are a professional IT documentation translator.
Translate the input text into Russian.

CRITICAL — Placeholders: Do NOT translate or modify placeholders like [CODE_BLOCK_N], [CLI_COMMAND_N], or [VAR_NAME_N] in any way. Copy them character-for-character including the exact number suffix. For example, [CODE_BLOCK_0] must remain [CODE_BLOCK_0], never [CODE_BLOCK_1] or any other variant.

CRITICAL — Formatting: Preserve the original structure exactly. Blank lines between paragraphs, line breaks within text, bullet points, numbered lists, and indentation must all appear in the translation at the same positions as in the source.

Adhere strictly to the provided glossary terms if any are listed in the input.
Output the translated text inside the translated_text field.
""",
    output_schema=TranslationOutput,
    output_key="translated_text",
)


# 3b. grammar_corrector — native-speaker grammar pass: fixes case, declension, agreement.
grammar_corrector = LlmAgent(
    name="grammar_corrector",
    model=Gemini(model=MODEL_NAME),
    instruction="""You are a native Russian speaker and professional editor.
You receive a Russian translation of an IT document and must correct grammatical errors AND unnatural word choices.

Fix grammatical errors:
- Wrong case (падеж): e.g. "статус ветка" → "статус ветки"
- Wrong declension of nouns, adjectives, pronouns
- Wrong verb form or aspect
- Gender agreement errors

Fix unnatural calques — replace heavy transliterations that sound awkward to a native IT professional:
- "гранулярность" → "детализация" or "уровень детализации"
- "имплементация" → "реализация" or "внедрение"
- "валидировать" → "проверять"
- "ресолвить" → "разрешать"
- Other direct transliterations that a native speaker would never write in professional documentation

Rules:
- Do NOT change correct terminology from the glossary.
- Do NOT add or remove sentences.
- Do NOT alter placeholders like [CODE_BLOCK_N], [CLI_COMMAND_N], [VAR_NAME_N] — copy them character-for-character.
- Do NOT change the structure: preserve blank lines, line breaks, lists, and indentation.
- If nothing needs correction, return the text unchanged.

Output the corrected Russian text in the translated_text field.
""",
    output_schema=GrammarCorrectorOutput,
    output_key="translated_text",
)


# 4. Reviewer Preparation Node
@node
def prepare_reviewer_input(ctx: Context, node_input: GrammarCorrectorOutput) -> Event:
    parsed_data = ctx.state.get("parsed_data")
    parsed = (
        ParserOutput(**parsed_data)
        if parsed_data
        else ParserOutput(document_text="", protected_blocks={})
    )

    glossary_terms = ctx.state.get("glossary_terms", [])
    glossary_str = (
        "Glossary terms used:\n"
        + "\n".join(f"- {t['english']} -> {t['russian']}" for t in glossary_terms)
        if glossary_terms
        else "No glossary terms were matched."
    )

    prompt = f"""Analyze the English source text and its Russian translation.

{glossary_str}

English source text (with placeholders):
{parsed.document_text}

Russian translation (with placeholders):
{node_input.translated_text}

Task:
1. Verify that all placeholders (e.g. [CODE_BLOCK_N], [CLI_COMMAND_N], [VAR_NAME_N]) are exactly preserved and unaltered in the Russian translation.
2. Check for translation consistency against the provided glossary terms.
3. Identify and flag any English terms in the source text that are domain-specific IT terms, are not in the glossary, and might be ambiguous or require a standard/agreed Russian translation.
"""
    return Event(
        output=PrepareReviewerInputOutput(prompt=prompt),
        actions=EventActions(
            state_delta={"translated_text": node_input.translated_text}
        ),
    )


# 4. reviewer — checks the translation and flags any ambiguous terms.
reviewer = LlmAgent(
    name="reviewer",
    model=Gemini(model=MODEL_NAME),
    instruction="""You are a translation quality reviewer.
Analyze the source and translation comparison provided in the input.
Perform the checks:
1. Are all placeholders ([CODE_BLOCK_N], [CLI_COMMAND_N], [VAR_NAME_N]) identical and unaltered in the translation?
2. Are the glossary terms followed consistently?
3. Are there IT terms in the English source text that have MULTIPLE competing accepted Russian translations in common professional use — where the choice reflects a real stylistic/organizational convention? Flag ONLY those.

IMPORTANT — do NOT flag:
- Terms with a single universally accepted transliteration (e.g. webhook→вебхук, cache→кэш, token→токен, API→API, JSON→JSON, HTTP→HTTP).
- Proper nouns, product names, brand names.
- Terms already resolved by the glossary.
- Terms where the translator's choice is clearly correct and uncontested.

ONLY flag terms like: commit (коммит vs фиксация), deploy (деплой vs развёртывание), endpoint (эндпоинт vs конечная точка), merge (мёрж vs слияние) — where two or more competing standard translations are in active professional use and the organization may have a preference.

Output:
- is_valid: true only if ALL placeholders are intact, glossary is followed, and there are NO genuinely ambiguous terms needing human review.
- feedback: explanation of your review findings.
- ambiguous_terms: list of English terms (max 3) that are genuinely contested. If is_valid is true, this list must be empty.
""",
    output_schema=ReviewerOutput,
    output_key="review_results",
)




# Final Node: reassembler — restores protected blocks.
@node
def reassembler(ctx: Context, node_input: Any) -> Event:
    parsed_data = ctx.state.get("parsed_data")
    translated_text = ctx.state.get("translated_text", "")

    if not parsed_data:
        return Event(
            output=translated_text,
            content=types.Content(
                role="model", parts=[types.Part.from_text(text=translated_text)]
            ),
        )

    parsed = ParserOutput(**parsed_data)
    final_text = translated_text

    # First pass: exact match
    unmatched: Dict[str, str] = {}
    for placeholder, original in parsed.protected_blocks.items():
        pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
        if pattern.search(final_text):
            final_text = pattern.sub(lambda _, o=original: o, final_text)
        else:
            unmatched[placeholder] = original

    # Second pass: fuzzy fallback when LLM shifted the index (e.g. [CODE_BLOCK_0] → [CODE_BLOCK_1])
    still_unmatched: Dict[str, str] = {}
    for placeholder, original in unmatched.items():
        ph_match = re.match(r'\[([A-Z_]+)_\d+\]', placeholder, re.IGNORECASE)
        if ph_match:
            block_type = re.escape(ph_match.group(1))
            fuzzy = re.compile(r'\[' + block_type + r'_\d+\]', re.IGNORECASE)
            if fuzzy.search(final_text):
                final_text = fuzzy.sub(lambda _, o=original: o, final_text, count=1)
                continue
        still_unmatched[placeholder] = original

    # Third pass: universal positional fallback when LLM also changed the placeholder TYPE
    # (e.g. [CLI_COMMAND_0] → [CODE_BLOCK_0]). Match orphaned tokens in the text to
    # unmatched originals in document order.
    if still_unmatched:
        known_keys_upper = {k.upper() for k in parsed.protected_blocks}
        universal = re.compile(r'\[[A-Z_]+_\d+\]', re.IGNORECASE)
        orphaned = [
            m.group(0) for m in universal.finditer(final_text)
            if m.group(0).upper() not in known_keys_upper
        ]
        for orphan, original in zip(orphaned, still_unmatched.values()):
            final_text = final_text.replace(orphan, original, 1)

    return Event(
        output=final_text,
        actions=EventActions(state_delta={"final_text": final_text}),
    )


# tooltip_generator — one batch Gemini call to produce 1-sentence explanations for all matched terms.
@node
def tooltip_generator(ctx: Context, node_input: Any) -> Event:
    glossary_terms = ctx.state.get("glossary_terms", [])

    # Explain all non-trivial terms: skip only pure acronyms kept as-is (API, JSON, HTTP, etc.)
    # and very short tokens (CI, CD). Single-word transliterations like "биллинг", "квота" are included.
    candidates = [
        t for t in glossary_terms
        if t["russian"] != t["english"]   # kept as-is → no tooltip needed
        and len(t["english"]) > 3         # skip CI, CD, etc.
        and not t["english"].isupper()    # skip pure acronyms (URL, SDK, CLI...)
    ][:12]

    tooltip_data: Dict[str, str] = {}
    if candidates:
        terms_list = "\n".join(
            f'- "{t["english"]}" (по-русски: "{t["russian"]}")'
            for t in candidates
        )
        batch_prompt = (
            "For each IT term below, write one short Russian sentence (under 100 chars) "
            "explaining what it means in software engineering context. "
            "Return ONLY a JSON object mapping the Russian term to its explanation.\n"
            'Example: {"балансировщик нагрузки": "Компонент, распределяющий запросы между серверами."}\n\n'
            f"Terms:\n{terms_list}"
        )
        try:
            raw = _gemini_call(batch_prompt)
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                tooltip_data = json.loads(m.group(0))
        except Exception:
            pass

    return Event(
        output=node_input,
        actions=EventActions(state_delta={"tooltip_data": tooltip_data}),
    )


# html_formatter — converts the final translated text to a styled HTML document with tooltips
# and writes it to output.html next to the project root.
@node
def html_formatter(ctx: Context, node_input: Any) -> Event:
    final_text = ctx.state.get("final_text", "")
    tooltip_data = ctx.state.get("tooltip_data", {})

    content = _text_to_html(final_text, tooltip_data)
    html_doc = _HTML_TEMPLATE.format(content=content)

    try:
        with open(_OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html_doc)
        msg = f"✅ Перевод готов. Открой файл output.html в браузере для просмотра с подсказками к терминам."
    except Exception as e:
        msg = f"⚠️ Не удалось сохранить output.html: {e}\n\n{final_text}"

    return Event(
        output=msg,
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


# --- Workflow & App Setup ---

root_agent = Workflow(
    name="translation_workflow",
    edges=[
        (START, parser),
        (parser, glossary_lookup),
        (glossary_lookup, auto_glossary_fill),
        (auto_glossary_fill, prepare_translator_input),
        (prepare_translator_input, translator),
        (translator, grammar_corrector),
        (grammar_corrector, prepare_reviewer_input),
        (prepare_reviewer_input, reviewer),
        (reviewer, reassembler),
        (reassembler, tooltip_generator),
        (tooltip_generator, html_formatter),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
