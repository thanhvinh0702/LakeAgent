from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lake_agent.config import LLMSettings
from lake_agent.qa.agents import SupervisorAgent, ModalitySubagent, AnswerVerifierAgent
from lake_agent.qa.retriever import CrossRetriever


class QAPipelineTest(unittest.TestCase):
    def test_supervisor_routes_correctly(self) -> None:
        settings = LLMSettings(api_key="fake-key", model_name="fake-model")
        supervisor = SupervisorAgent(settings)
        
        # Test routing fallback logic
        routing = supervisor.route_query("Which image contains the blue digit?")
        self.assertIn("image", routing.modalities)
        
        routing_tab = supervisor.route_query("correlation between Limit and Balance in Credit.csv")
        self.assertIn("tabular", routing_tab.modalities)

    def test_cross_retriever_filesystem_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            datalake_dir = Path(temp_dir)
            (datalake_dir / "notes.txt").write_text("Hello project info", encoding="utf-8")
            (datalake_dir / "Credit.csv").write_text("Limit,Balance\n100,50\n", encoding="utf-8")
            
            retriever = CrossRetriever(datalake_dir=str(datalake_dir))
            
            # Text query
            text_ctx = retriever.retrieve("text", "notes.txt")
            self.assertEqual(1, len(text_ctx))
            self.assertEqual("notes.txt", text_ctx[0]["filename"])
            self.assertIn("Hello project info", text_ctx[0]["content"])
            
            # Tabular query
            tab_ctx = retriever.retrieve("tabular", "Credit.csv")
            self.assertEqual(1, len(tab_ctx))
            self.assertEqual("Credit.csv", tab_ctx[0]["filename"])
            self.assertIn("Limit, Balance", tab_ctx[0]["content"])

    @patch("lake_agent.qa.agents.init_chat_model")
    def test_answer_verifier_resolves_discrepancies(self, mock_init_chat_model: MagicMock) -> None:
        # Mock model response for structured verification
        mock_model = MagicMock()
        mock_init_chat_model.return_value = mock_model
        
        # Mock structured output invocation
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        
        from lake_agent.qa.agents import VerificationOutput
        mock_structured.invoke.return_value = VerificationOutput(
            answer="125",
            evidences=["workshop_03.22.m4a"]
        )
        
        settings = LLMSettings(api_key="fake-key", model_name="fake-model")
        verifier = AnswerVerifierAgent(settings)
        
        result = verifier.verify_answer(
            question="What is the total number of workshop participants?",
            proposed_answers=[{
                "answer": "125 total participants",
                "evidences": ["workshop_03.22.m4a"]
            }]
        )
        
        self.assertEqual("125", result.answer)
        self.assertEqual(["workshop_03.22.m4a"], result.evidences)


if __name__ == "__main__":
    unittest.main()
