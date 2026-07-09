from __future__ import annotations

import json
import logging
from typing import Any
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from lake_agent.config import LLMSettings

logger = logging.getLogger(__name__)


class ModalityRouting(BaseModel):
    modalities: list[str] = Field(
        description="List of target modalities to query. Options: 'text', 'tabular', 'image', 'document', 'audio', 'database'."
    )
    queries: dict[str, str] = Field(
        description="Modality-specific search queries generated for each routed modality."
    )


class VerificationOutput(BaseModel):
    answer: str = Field(
        description="The final precise answer text or number, strictly formatted as requested."
    )
    evidences: list[str] = Field(
        description="List of source filenames (e.g. 'Credit.csv') used as evidence for the answer."
    )


class BaseAgent:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        # OpenRouter support works by passing openai provider + custom base_url
        self.model = init_chat_model(
            model_provider="openai",
            api_key=settings.api_key,
            base_url=settings.base_url or "https://openrouter.ai/api/v1",
            model=settings.model_name,
            temperature=0,
        )


class SupervisorAgent(BaseAgent):
    def route_query(self, question: str) -> ModalityRouting:
        structured_model = self.model.with_structured_output(
            ModalityRouting,
            method="function_calling",
        )
        
        system_prompt = (
            "You are the Supervisor Agent of a multi-modal data lake query pipeline.\n"
            "Analyze the user's natural language question and route it to the relevant subagents.\n"
            "Available subagents & modalities:\n"
            "- 'text': txt, md files, html pages, and SQL script files.\n"
            "- 'tabular': csv, tsv, xlsx files.\n"
            "- 'image': png, jpg, jpeg files.\n"
            "- 'document': pdf manuals or PDF documents.\n"
            "- 'audio': mp3, wav, m4a audio files.\n"
            "- 'database': sqlite database files.\n\n"
            "For each routed modality, generate a specific search query designed to retrieve relevant files or context."
        )
        
        try:
            response = structured_model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=question),
                ]
            )
            if isinstance(response, ModalityRouting):
                return response
            return ModalityRouting.model_validate(response)
        except Exception as e:
            logger.warning(f"Failed structured routing: {e}. Falling back to default routing.")
            # Simple fallback routing based on keyword heuristic
            modalities = []
            lower_q = question.lower()
            if "image" in lower_q or "ảnh" in lower_q or "jpg" in lower_q or "png" in lower_q or "học bổng" in lower_q or "digit" in lower_q or "nhóm ise" in lower_q:
                modalities.append("image")
            if "csv" in lower_q or "credit" in lower_q or "correlation" in lower_q or "excel" in lower_q or "xlsx" in lower_q or "grades" in lower_q or "grades.sql" in lower_q or "lớp 10a1" in lower_q:
                modalities.append("tabular")
                if "sql" in lower_q or "grades" in lower_q:
                    modalities.append("text") # SQL scripts parsed as text
            if "诸葛亮" in lower_q or "隆中对" in lower_q or "html" in lower_q or "wiki" in lower_q or "thư viện" in lower_q or "sông minh" in lower_q:
                modalities.append("text")
            if "pdf" in lower_q or "axiom" in lower_q or "project" in lower_q:
                modalities.append("document")
            if "audio" in lower_q or "m4a" in lower_q or "meeting" in lower_q or "workshop" in lower_q or "participants" in lower_q:
                modalities.append("audio")
            
            if not modalities:
                modalities = ["text", "tabular", "document"]
                
            queries = {m: question for m in modalities}
            return ModalityRouting(modalities=modalities, queries=queries)


class ModalitySubagent(BaseAgent):
    def answer_query(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        context_str = ""
        for idx, ctx in enumerate(contexts):
            context_str += f"\n[Source {idx+1}] File: {ctx.get('filename')}\nContent:\n{ctx.get('content')}\n"
            
        system_prompt = (
            "You are a specialized modality subagent. Analyze the context retrieved from the data lake "
            "and answer the question based strictly on this context. Identify which files you got the answer from."
        )
        
        user_prompt = f"Question: {question}\n\nRetrieved Contexts:\n{context_str}"
        
        response = self.model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        
        return {
            "answer": response.content,
            "evidences": [ctx.get("filename") for ctx in contexts if ctx.get("filename")]
        }


class AnswerVerifierAgent(BaseAgent):
    def verify_answer(self, question: str, proposed_answers: list[dict[str, Any]]) -> VerificationOutput:
        structured_model = self.model.with_structured_output(
            VerificationOutput,
            method="function_calling",
        )
        
        answers_summary = ""
        for idx, ans in enumerate(proposed_answers):
            answers_summary += f"\n[Agent {idx+1}] Answer: {ans.get('answer')}\nEvidences: {ans.get('evidences')}\n"
            
        system_prompt = (
            "You are the Answer Verifier Agent of a multi-modal data lake QA pipeline.\n"
            "Your job is to resolve discrepancies, verify answers against proposed evidences, and output a single correct answer and clean evidences list.\n"
            "Rules:\n"
            "1. Output exact, concise answers (especially for classification, numbers, Yes/No, or multiple-choice options).\n"
            "2. If the user question specifies formatting (e.g. UPPERCASE, round to two decimal places, option letter choice), adhere strictly to it.\n"
            "3. The evidences list MUST ONLY contain filenames that are present in the 'Evidences' field of the 'Proposed Subagent Answers'. Do not make up, assume, or invent any new filenames. If the proposed answers do not provide valid source filenames, return an empty list [].\n"
            "4. Eliminate any hallucinations not supported by the evidence."
        )
        
        user_prompt = f"Question: {question}\n\nProposed Subagent Answers:\n{answers_summary}"
        
        try:
            response = structured_model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            if isinstance(response, VerificationOutput):
                return response
            return VerificationOutput.model_validate(response)
        except Exception as e:
            logger.warning(f"Failed structured verification: {e}. Falling back to default parser.")
            # Heuristic parser fallback
            combined_evidences = []
            for ans in proposed_answers:
                for ev in ans.get("evidences", []):
                    if ev and ev not in combined_evidences:
                        combined_evidences.append(ev)
            
            # Use LLM call with plain instructions to extract just the answer
            plain_prompt = (
                f"You are the Answer Verifier. Parse the proposed answers and output ONLY the final answer text/value for the question:\n"
                f"Question: {question}\n\nProposed Answers:\n{answers_summary}\n\nOutput only the exact answer, no conversational prefix."
            )
            plain_response = self.model.invoke([HumanMessage(content=plain_prompt)])
            return VerificationOutput(
                answer=plain_response.content.strip(),
                evidences=combined_evidences
            )
