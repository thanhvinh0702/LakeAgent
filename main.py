from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from deepagents.backends import LocalShellBackend
from deepagents.middleware import FilesystemMiddleware, SummarizationMiddleware

from lake_agent.config import LocalSettings
from lake_agent.tools import IndexedDataRetriever, build_langchain_retrieval_tools

PROJECT_ROOT = Path(__file__).resolve().parent

SYSTEM_PROMPT_TEMPLATE = """
You are LakeAgent, a retrieval-first analyst for a multi-modal data lake.

Your job is to answer user questions accurately. Use the indexed retrieval tools first to find the
right files and passages quickly, then continue investigating until the evidence is strong enough to
answer with confidence.

Data lake:
- `DATALAKE_DIR` from `.env`: `{datalake_dir}`
- Resolved data lake root: `{resolved_datalake_dir}`
- Repository root: `{project_root}`

Core workflow:
1. Start with retrieval tools, not full-file reading.
2. If the query does not clearly specify the modality of the underlying data, you must start with
   `search_all_indexed_data`.
3. Do not start with modality-specific search tools unless the question clearly points to one modality
   or broad search results make the relevant modality clear.
4. Prefer modality-specific search tools when the question clearly points to one modality:
   - `search_tabular_data` for tables/spreadsheets/CSV-style facts
   - `search_text_data` for TXT/MD narrative text
   - `search_document_data` for PDF/DOCX/DOC/RTF documents
   - `search_slideshow_data` for PPT/PPTX slides
   - `search_image_data` for images, OCR text, or image summaries
5. If the first tool results are not enough to answer confidently, continue investigating.
6. Investigation ladder when evidence is still insufficient:
   - page further with the same search tool using larger `offset`
   - try another relevant search tool if the first modality may be incomplete
   - use `get_indexed_file_summary` for promising file paths
   - inspect local file content around the returned `position`
   - read a larger excerpt or the full file when the answer requires full-document certainty
7. Use `get_indexed_file_summary` when you need file-level understanding before opening raw content.
8. Do not read an entire local file into context unless smaller targeted inspection is still not
   enough to answer correctly.
9. Only inspect local files after retrieval has identified a promising file or the user explicitly
    asks for direct file reading.
10. When reading a file locally, read only a small window around the retrieved location first:
   - line-based hits: inspect only nearby lines around `position.start`/`position.end`
   - page-based hits: inspect only the relevant page region or a narrow extracted excerpt if possible
   - slide-based hits: inspect only the relevant slide or neighboring slide
11. Escalate to broader excerpts or full-file reading when targeted inspection still leaves material
    uncertainty.

Path rules:
- For `execute`, use `execution_file_path` when available.
- Filesystem tool paths are normalized automatically to the data lake virtual root.
- For `ls` and `read_file`, just use the natural data-lake path you have.
- Do not pass host absolute paths to `execute` unless a command truly requires them.
- In final answers, prefer data-lake-relative paths such as `/folder/file.ext`.
- Never present host filesystem paths such as `/data/Data-Lake/...` or `/home/...` in the final
  answer.

Use retrieved results like this:
- `content` is the main evidence.
- `position` tells you where to inspect next if needed.
- `score` is only a relevance hint.
- Cross-check multiple hits for comparison, aggregation, conflicts, or high-confidence claims.

Answering rules:
- Ground every answer in retrieved evidence.
- Do not use background knowledge, world knowledge, or unstated memory as a substitute for missing
  evidence from the data lake.
- Only give a final answer when the collected evidence is strong enough for the specific question.
- If the evidence is incomplete or conflicting, say so plainly.
- If the current tool results are not enough, search more or inspect the relevant file before
  concluding.
- If you are still uncertain after reasonable investigation, do not give a tentative answer.
- If there is not enough data, answer exactly: `Not enough data to answer.`
- Prefer `Not enough data to answer.` over a weak, guessed, partial, or temporary conclusion.
- Keep the final answer concise.
- Do not expose chain-of-thought or internal reasoning.

Important discipline:
- Retrieval first.
- Use `search_all_indexed_data` first whenever modality is unclear.
- Keep investigating until evidence is sufficient.
- Page more before broad raw file reads.
- Targeted inspection before full-file reading.
- Full-file reading is allowed when required for a reliable answer.
""".strip()

FINAL_ANSWER_VERIFIER_PROMPT = """
You rewrite an already-computed answer into the exact final output format required by the user.

Rules:
- Preserve the meaning of the draft answer.
- Preserve any file path or image path that is necessary for the user to open, view, or identify
  the referenced asset.
- Output only the final answer text.
- Do not add explanations, labels, bullets, markdown, code fences, or citations unless the user
  explicitly requested them.
- If the user asked to show, display, open, or provide an image or file, the final answer should
  retain the relevant file path.
- If the user only asked to see, show, view, open, or provide an image or file, do not add a
  description of the asset unless the user explicitly asked for one.
- Prefer natural plain text over markdown when plain text is sufficient.
- Do not introduce markdown image syntax or file-link syntax unless the user explicitly asked for
  markdown or the draft already must stay in that format.
- When referring to an image or file, prefer readable phrasing such as:
  `Anh cac thanh vien cua nhom iSE nam trong file "definitely-100-percent-not-ise-members-image.png".`
- Optimize for exact-match grading.
- If the user asked for an exact format, exact token, exact number, exact option label, or a single
  final string, output exactly that and nothing else.
- If the correct answer is a single number, output only that number.
- If the correct answer is a short phrase or entity, output only that phrase or entity.
- Remove wrapper phrases such as "The answer is", "There are", "It is", "The result is", or any
  similar explanatory wording.
- If the draft answer is `Not enough data to answer.`, return it unchanged.
- Return exactly `Not enough data to answer.` only when the draft answer clearly concludes that the
  question cannot be answered reliably from the data lake.
- Do not return `Not enough data to answer.` merely because the draft says there is no single
  dedicated table, no one-to-one source, or no directly precomputed field for the answer.
- If the draft answer provides a supported answer by combining evidence across multiple retrieved
  sources, keep that answer even if the draft also notes limitations, indirect evidence, or missing
  dedicated tables.
- Do not preserve or restate answers that the draft explicitly labels as based on general knowledge,
  background knowledge, or information outside the data lake.
- Fix formatting drift, extra words, prefixes, suffixes, or commentary.
- When in doubt, make the output shorter and stricter.

Examples:
- Query: "Return only the number of significant genes."
  Draft: "There are 16 significant genes identified by acetylproteomics."
  Output: 16
- Query: "Answer with Yes or No only."
  Draft: "Yes, the table shows a positive correlation."
  Output: Yes
- Query: "Cho tôi xem ảnh đó"
  Draft: "Đây là ảnh được lưu trong data lake: /folder/example-image.jpg. Mô tả: ..."
  Output: "Ảnh bạn cần nằm trong file \"/folder/example-image.jpg\"."
""".strip()

_FILESYSTEM_TOOL_NAMES = {
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a retrieval-first agent questions about the indexed data lake.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="User question to ask the agent. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming and print only the final agent response.",
    )
    return parser


def resolve_datalake_dir(datalake_dir: str) -> Path:
    candidate = Path(datalake_dir).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def build_backend(resolved_datalake_dir: Path) -> LocalShellBackend:
    return LocalShellBackend(
        root_dir=str(resolved_datalake_dir),
        virtual_mode=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "DATALAKE_DIR": str(resolved_datalake_dir),
        },
    )


def build_model():
    return init_chat_model(
        model=os.getenv("AGENT_MODEL_NAME"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_verifier_model():
    return init_chat_model(
        model=os.getenv("AGENT_MODEL_NAME"),
        model_provider="openai",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def build_system_prompt(
    datalake_dir: str,
    *,
    resolved_datalake_dir: Path,
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        datalake_dir=datalake_dir,
        project_root=str(PROJECT_ROOT),
        resolved_datalake_dir=str(resolved_datalake_dir),
    )


class FilesystemPathNormalizationMiddleware(AgentMiddleware):
    def __init__(self, datalake_root: Path) -> None:
        self._datalake_root = datalake_root.resolve()

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call.get("name")
        if tool_name not in _FILESYSTEM_TOOL_NAMES:
            return handler(request)

        args = request.tool_call.get("args", {})
        if not isinstance(args, dict):
            return handler(request)

        normalized_args = dict(args)
        for key in ("path", "file_path"):
            value = normalized_args.get(key)
            if isinstance(value, str):
                normalized_args[key] = self._normalize_filesystem_path(value)

        modified_call = {
            **request.tool_call,
            "args": normalized_args,
        }
        return handler(request.override(tool_call=modified_call))

    def _normalize_filesystem_path(self, raw_path: str) -> str:
        value = raw_path.strip()
        if not value:
            return value
        if value == ".":
            return "/"
        if value.startswith("/"):
            host_prefix = self._datalake_root.as_posix()
            if value == host_prefix:
                return "/"
            if value.startswith(host_prefix + "/"):
                suffix = value[len(host_prefix) :].replace("\\", "/")
                return suffix or "/"
            return value
        normalized = value.replace("\\", "/").lstrip("./")
        return "/" if not normalized else f"/{normalized}"


def verify_final_answer(query: str, draft_answer: str, model) -> str:
    response = model.invoke(
        [
            SystemMessage(content=FINAL_ANSWER_VERIFIER_PROMPT),
            HumanMessage(
                content=(
                    "User query:\n"
                    f"{query}\n\n"
                    "Draft answer:\n"
                    f"{draft_answer}\n\n"
                    "Return the final answer only."
                )
            ),
        ]
    )
    content = getattr(response, "text", None)
    if content is None:
        content = getattr(response, "content", "")
    if isinstance(content, list):
        normalized = "".join(str(part) for part in content).strip()
    else:
        normalized = str(content).strip()
    return normalized or "Not enough data to answer."


def extract_final_text(response: dict) -> str:
    messages = response.get("messages", [])
    if not messages:
        return ""
    final_message = messages[-1]
    content = getattr(final_message, "text", None)
    if content is None:
        content = getattr(final_message, "content", "")
    if isinstance(content, list):
        return "".join(str(part) for part in content).strip()
    return str(content).strip()

def build_agent_and_retriever():
    model = build_model()
    verifier_model = build_verifier_model()
    local_settings = LocalSettings.from_env()
    resolved_datalake_dir = resolve_datalake_dir(local_settings.datalake_dir)
    backend = build_backend(resolved_datalake_dir)
    retriever = IndexedDataRetriever.from_env()
    tools = build_langchain_retrieval_tools(retriever)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(
            local_settings.datalake_dir,
            resolved_datalake_dir=resolved_datalake_dir,
        ),
        middleware=[
            FilesystemPathNormalizationMiddleware(resolved_datalake_dir),
            FilesystemMiddleware(backend=backend),
            SummarizationMiddleware(model=model, backend=backend),
            TodoListMiddleware(),
        ],
        name="lakeagent_retrieval_agent",
    )
    return agent, retriever, model, verifier_model


def run_streaming(agent, query: str, verifier_model) -> None:
    stream = agent.stream_events(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
        version="v3",
    )

    if hasattr(stream, "interleave"):
        for name, item in stream.interleave("messages", "tool_calls"):
            if name == "messages":
                print(f"\n[{item.node.upper()}]")
                for delta in item.text:
                    print(delta, end="", flush=True)
            elif name == "tool_calls":
                print("\n\n" + "=" * 80)
                print("[TOOL CALL]")
                print(f"TOOL: {item.tool_name}")
                print(f"INPUT: {item.input}")
                for delta in item.output_deltas:
                    print(delta, end="", flush=True)
                if item.error:
                    print(f"\nERROR: {item.error}")
                else:
                    print(f"\nOUTPUT: {item.output}")
    else:
        for message in stream.messages:
            print(f"\n[{message.node.upper()}]")
            for delta in message.text:
                print(delta, end="", flush=True)

        tool_calls = getattr(stream, "tool_calls", [])
        for item in tool_calls:
            print("\n\n" + "=" * 80)
            print("[TOOL CALL]")
            print(f"TOOL: {item.tool_name}")
            print(f"INPUT: {item.input}")
            for delta in item.output_deltas:
                print(delta, end="", flush=True)
            if item.error:
                print(f"\nERROR: {item.error}")
            else:
                print(f"\nOUTPUT: {item.output}")

    final_state = stream.output or {}
    draft_answer = extract_final_text(final_state)
    print("\n\n[AGENT FINAL]")
    print(draft_answer)
    verified_answer = verify_final_answer(query, draft_answer, verifier_model)
    print("\n\n[VERIFIED FINAL]")
    print(verified_answer)


def run_non_streaming(agent, query: str, verifier_model) -> None:
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )
    draft_answer = extract_final_text(response)
    verified_answer = verify_final_answer(query, draft_answer, verifier_model)
    print(verified_answer)


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    query = args.query or input("Question: ").strip()
    if not query:
        print("A non-empty question is required.")
        return 1

    agent, retriever, model, verifier_model = build_agent_and_retriever()
    try:
        if args.no_stream:
            run_non_streaming(agent, query, verifier_model)
        else:
            run_streaming(agent, query, verifier_model)
    finally:
        retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
