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
from typing import List, Dict, Optional, Any, AsyncGenerator, Union
from pydantic import BaseModel, Field

import google.auth
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, START, node
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
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


class ParserOutput(BaseModel):
    document_text: str = Field(
        description="The parsed document text with unique placeholders in the format {{CODE_BLOCK_N}}, {{CLI_COMMAND_N}}, or {{VAR_NAME_N}}."
    )
    protected_blocks: Dict[str, str] = Field(
        description="Mapping of placeholders (e.g. '{{CODE_BLOCK_0}}') to their original content."
    )


class GlossaryMatch(BaseModel):
    english: str
    russian: str


class GlossaryLookupOutput(BaseModel):
    matching_terms: List[GlossaryMatch]


class PrepareTranslatorInputOutput(BaseModel):
    prompt: str


class TranslationOutput(BaseModel):
    translated_text: str


class PrepareReviewerInputOutput(BaseModel):
    prompt: str


class ReviewerOutput(BaseModel):
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


class TermLookupOutput(BaseModel):
    term: str
    resolved: bool
    wikipedia_url: Optional[str] = None
    translation: str


# --- Node Implementations ---

# 1. parser — extracts the document text, identifying and separating code blocks, CLI commands, and variable names so they are never translated.
parser = LlmAgent(
    name="parser",
    model=Gemini(model=MODEL_NAME),
    instruction="""You are a professional technical document parser.
Analyze the input English document. Identify and extract:
1. Code blocks (e.g., lines starting and ending with triple backticks ```, or inline code snippets in single backticks containing more than 2 words).
2. CLI commands (e.g., commands starting with CLI patterns or running commands in code blocks or inline).
3. Variable names / identifiers / class names (e.g., `user_id`, `resumability_config`, `get_weather`).

Replace each identified element with a unique placeholder in the text in the format:
- {{CODE_BLOCK_N}} for code blocks
- {{CLI_COMMAND_N}} for CLI commands
- {{VAR_NAME_N}} for variable names
where N is a 0-based counter.

Output both:
- document_text: the modified document text containing placeholders.
- protected_blocks: a dictionary mapping each placeholder to its exact original, unaltered content.
""",
    output_schema=ParserOutput,
    output_key="parsed_data",
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
        pattern = r"\b" + re.escape(eng.lower()) + r"\b"
        if re.search(pattern, text_lower):
            matching_terms.append(GlossaryMatch(english=eng, russian=rus))

    return Event(
        output=GlossaryLookupOutput(matching_terms=matching_terms),
        actions=EventActions(
            state_delta={
                "parsed_data": node_input.model_dump(),
                "glossary_terms": [m.model_dump() for m in matching_terms],
            }
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
        glossary_str = "GLOSSARY TERMS:\n" + "\n".join(
            f"- {t.english} -> {t.russian}" for t in node_input.matching_terms
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
Do not translate placeholders like {{CODE_BLOCK_N}}, {{CLI_COMMAND_N}}, or {{VAR_NAME_N}}—leave them exactly as they are.
Adhere strictly to the provided glossary terms if any are listed in the input.
Output the translated text inside the translated_text field.
""",
    output_schema=TranslationOutput,
    output_key="translated_text",
)


# 4. Reviewer Preparation Node
@node
def prepare_reviewer_input(ctx: Context, node_input: TranslationOutput) -> Event:
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
1. Verify that all placeholders (e.g. {{CODE_BLOCK_N}}, {{CLI_COMMAND_N}}, {{VAR_NAME_N}}) are exactly preserved and unaltered in the Russian translation.
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
1. Are all placeholders ({{CODE_BLOCK_N}}, {{CLI_COMMAND_N}}, {{VAR_NAME_N}}) identical and unaltered in the translation?
2. Are the glossary terms followed consistently?
3. Are there any other specific IT terms in the English source text that are not in the glossary, and are ambiguous or need a human translator to confirm or provide their Russian translation?

Output:
- is_valid: true only if ALL placeholders are intact, glossary is followed, and there are NO ambiguous terms needing human review.
- feedback: explanation of your review findings.
- ambiguous_terms: list of English terms (max 3) that are new, ambiguous, or need human review/clarification. If is_valid is true, this list must be empty.
""",
    output_schema=ReviewerOutput,
    output_key="review_results",
)


# 5. A human-in-the-loop RequestInput node that pauses when the reviewer flags an ambiguous term.
@node(rerun_on_resume=True)
async def human_review(
    ctx: Context, node_input: Any
) -> AsyncGenerator[Union[Event, RequestInput], None]:
    review_results_data = ctx.state.get("review_results")
    if not review_results_data:
        yield Event(output="skip", actions=EventActions(route="skip"))
        return

    review_results = ReviewerOutput(**review_results_data)
    if review_results.is_valid or not review_results.ambiguous_terms:
        yield Event(output="skip", actions=EventActions(route="skip"))
        return

    resolved_terms = ctx.state.get("resolved_terms", {})
    term_to_resolve = None
    for term in review_results.ambiguous_terms:
        if term not in resolved_terms:
            term_to_resolve = term
            break

    if not term_to_resolve:
        yield Event(output="resolved", actions=EventActions(route="resolved"))
        return

    interrupt_id = f"resolve_{term_to_resolve.lower().replace(' ', '_')}"

    if ctx.resume_inputs and interrupt_id in ctx.resume_inputs:
        human_val = ctx.resume_inputs[interrupt_id].strip()
        route_val = "unknown" if human_val.lower() == "unknown" else "user_provided"
        yield Event(
            output=human_val,
            actions=EventActions(
                route=route_val,
                state_delta={
                    "current_ambiguous_term": term_to_resolve,
                    "human_response": human_val,
                },
            ),
        )
        return

    yield RequestInput(
        interrupt_id=interrupt_id,
        message=f"The reviewer flagged the term '{term_to_resolve}' as ambiguous/new. Please provide its Russian translation, or respond 'unknown' if you don't know.",
    )


# Helper function for Node 6 Wikipedia API lookup
def lookup_wikipedia(term: str) -> Optional[str]:
    encoded_term = urllib.parse.quote(term)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_term}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "AgenticTranslator/1.0 (contact@example.com)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("content_urls", {}).get("desktop", {}).get("page")
    except Exception:
        pass
    return None


# 6. term_lookup — looks up the term on Wikipedia if the human responds "unknown".
@node
def term_lookup(ctx: Context, node_input: str) -> Event:
    term = ctx.state.get("current_ambiguous_term")
    translated_text = ctx.state.get("translated_text", "")
    resolved_terms = ctx.state.get("resolved_terms", {})

    wiki_url = lookup_wikipedia(term)

    if wiki_url:
        footnote_num = len(resolved_terms) + 1
        footnote_marker = f"[^{footnote_num}]"

        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(translated_text):
            translated_text = pattern.sub(
                f"{term}{footnote_marker}", translated_text, count=1
            )
        else:
            translated_text += f" ({term}){footnote_marker}"

        translated_text += f"\n\n{footnote_marker}: {wiki_url}"

        resolved_terms[term] = {
            "term": term,
            "resolved": True,
            "wikipedia_url": wiki_url,
            "translation": f"{term} (Wikipedia)",
        }
    else:
        resolved_terms[term] = {
            "term": term,
            "resolved": False,
            "wikipedia_url": None,
            "translation": "unresolved",
        }

    return Event(
        output="lookup_done",
        actions=EventActions(
            state_delta={
                "translated_text": translated_text,
                "resolved_terms": resolved_terms,
            }
        ),
    )


# 7. glossary_update — persists decisions back into glossary.json.
@node
def glossary_update(ctx: Context, node_input: str) -> Event:
    term = ctx.state.get("current_ambiguous_term")
    resolved_terms = ctx.state.get("resolved_terms", {})

    try:
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            glossary = json.load(f)
    except Exception:
        glossary = {}

    human_response = ctx.state.get("human_response")
    if human_response and human_response.lower() != "unknown":
        glossary[term] = human_response
        resolved_terms[term] = {
            "term": term,
            "resolved": True,
            "wikipedia_url": None,
            "translation": human_response,
        }
    else:
        term_info = resolved_terms.get(term, {})
        translation = term_info.get("translation", "unresolved")
        glossary[term] = translation

    try:
        with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
            json.dump(glossary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving glossary: {e}")

    return Event(
        output="update_done",
        actions=EventActions(
            route="loop",
            state_delta={
                "resolved_terms": resolved_terms,
                "current_ambiguous_term": None,
                "human_response": None,
            },
        ),
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
    for placeholder, original in parsed.protected_blocks.items():
        pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
        final_text = pattern.sub(original, final_text)

    return Event(
        output=final_text,
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=final_text)]
        ),
    )


# Two thin bridge nodes so "skip" and "resolved" routes from human_review
# reach reassembler through distinct (from, to) pairs each.
# ADK graph deduplication is by (from_name, to_name) pair — route value is
# NOT considered — so we cannot send two routes to the same node directly.
@node
def finalize_skip(ctx: Context, node_input: Any) -> Any:
    """Bridge for 'skip' route: no ambiguous terms, proceed to final assembly."""
    return node_input


@node
def finalize_resolved(ctx: Context, node_input: Any) -> Any:
    """Bridge for 'resolved' route: all flagged terms handled, finalize."""
    return node_input


# --- Workflow & App Setup ---

root_agent = Workflow(
    name="translation_workflow",
    edges=[
        (START, parser),
        (parser, glossary_lookup),
        (glossary_lookup, prepare_translator_input),
        (prepare_translator_input, translator),
        (translator, prepare_reviewer_input),
        (prepare_reviewer_input, reviewer),
        (reviewer, human_review),
        # "skip" and "resolved" each go to their own bridge node first,
        # then unconditionally to reassembler — required because ADK
        # deduplicates edges by (from_name, to_name) ignoring route value.
        (human_review, {"skip": finalize_skip}),
        (human_review, {"resolved": finalize_resolved}),
        # Ambiguous term routes
        (human_review, {"unknown": term_lookup}),
        (human_review, {"user_provided": glossary_update}),
        # Bridge nodes → final assembly
        (finalize_skip, reassembler),
        (finalize_resolved, reassembler),
        # After lookup, update glossary and loop back to human_review
        (term_lookup, glossary_update),
        (glossary_update, {"loop": human_review}),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
