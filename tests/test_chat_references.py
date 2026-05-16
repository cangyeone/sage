"""Regression tests for chat reference grounding and visible skill sources."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask


PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from routes import chat as chat_routes


class TestChatReferences(unittest.TestCase):
    def test_bvalue_web_search_filters_irrelevant_records(self):
        papers = [
            {
                "source": "openalex",
                "title": "A Study on Rice Pest Insect and Disease Diagnosis Expert System Based on B/S Mode",
                "year": 2008,
                "authors": ["Unrelated Author"],
                "abstract": "A browser/server system for agricultural diagnosis.",
                "url": "https://example.test/rice",
            },
            {
                "source": "openalex",
                "title": "Maximum likelihood estimate of b in the formula log N = a - bM",
                "year": 1965,
                "authors": ["Keiiti Aki"],
                "abstract": "Earthquake magnitude-frequency b-value estimation for seismicity catalogs.",
                "url": "https://example.test/aki",
            },
        ]
        with patch("routes.chat._literature_web_search", return_value=papers) as search:
            ctx, sources = chat_routes._chat_web_search_context(
                {"enable_web_search": True, "web_max_results": 6},
                "给我一个 b 值的程序",
            )

        self.assertIn("earthquake b-value Gutenberg Richter", search.call_args.args[0])
        self.assertIn("Maximum likelihood estimate of b", ctx)
        self.assertNotIn("Rice Pest", ctx)
        self.assertEqual(len(sources), 1)
        self.assertIn("Maximum likelihood estimate of b", sources[0]["label"])
        self.assertEqual(sources[0]["url"], "https://example.test/aki")

    def test_code_draft_includes_bvalue_references_and_skill_sources(self):
        with patch("routes.chat.get_llm_config", return_value={"api_base": "http://llm", "model": "test"}), \
             patch(
                 "routes.chat._skill_context_with_sources",
                 return_value=(
                     "skill docs for b_value_analysis",
                     "",
                     [{"label": "[Skill] b_value_analysis: API guide", "url": "/api/chat/source_file?path=/tmp/SKILL.md"}],
                 ),
             ):
            messages, sources, _ = chat_routes._build_code_draft_messages(
                {"message": "给我一个 b 值的程序", "enable_web_search": False}
            )

        system = messages[0]["content"]
        self.assertIn("CANONICAL B-VALUE LITERATURE REFERENCES", system)
        self.assertIn("skill docs for b_value_analysis", system)
        self.assertTrue(any(src["label"].startswith("[Reference] Aki (1965)") for src in sources))
        self.assertTrue(any(src["url"].startswith("https://") for src in sources if src["label"].startswith("[Reference]")))
        self.assertIn(
            {"label": "[Skill] b_value_analysis: API guide", "url": "/api/chat/source_file?path=/tmp/SKILL.md"},
            sources,
        )

    def test_rag_qa_includes_skill_sources(self):
        with patch("routes.chat.get_llm_config", return_value={"api_base": "http://llm", "model": "test"}), \
             patch("routes.chat.get_kb_instance", return_value=None), \
             patch(
                 "routes.chat._skill_context_with_sources",
                 return_value=(
                     "skill docs for b_value_analysis",
                     "",
                     [{"label": "[Skill] b_value_analysis: API guide", "url": "/api/chat/source_file?path=/tmp/SKILL.md"}],
                 ),
             ):
            messages, sources, _ = chat_routes._build_rag_messages(
                {"message": "b 值怎么计算", "enable_web_search": False}
            )

        self.assertIn("skill docs for b_value_analysis", messages[0]["content"])
        self.assertIn(
            {"label": "[Skill] b_value_analysis: API guide", "url": "/api/chat/source_file?path=/tmp/SKILL.md"},
            sources,
        )

    def test_skill_sources_include_clickable_local_file_url(self):
        with patch("helpers.get_skill_loader") as loader_factory:
            loader = loader_factory.return_value
            loader.build_skill_context_with_rag.return_value = ("skill docs", "",)
            loader.search_skills.return_value = [{
                "name": "b_value_analysis",
                "description": "API guide",
                "path": str(PROJECT_ROOT / "seismo_skill" / "skills" / "b_value_analysis" / "SKILL.md"),
            }]

            _, _, sources = chat_routes._skill_context_with_sources("b 值")

        self.assertEqual(len(sources), 1)
        self.assertIn("[Skill] b_value_analysis", sources[0]["label"])
        self.assertTrue(sources[0]["url"].startswith("/api/chat/source_file?path="))

    def test_source_file_route_serves_allowed_skill_file(self):
        app = Flask(__name__)
        app.register_blueprint(chat_routes.bp)
        source_path = PROJECT_ROOT / "seismo_skill" / "skills" / "b_value_analysis" / "SKILL.md"

        with app.test_client() as client:
            resp = client.get(f"/api/chat/source_file?path={source_path}")

        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"b", resp.data.lower())


if __name__ == "__main__":
    unittest.main()
