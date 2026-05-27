"""Regression tests for temporary chat PDF upload."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import app as web_app
from state import _session_docs
from routes import chat as chat_routes


class TestChatUploadPdf(unittest.TestCase):
    def test_upload_uses_current_rag_extractors(self):
        session_id = "upload_regression"
        _session_docs.pop(session_id, None)

        pages = [(0, "This is a long enough extracted PDF page for temporary chat context.")]
        chunks = [{"page": 1, "text": pages[0][1]}]
        client = web_app.app.test_client()

        with patch("routes.chat._extract_session_pdf_chunks", return_value=(pages, chunks)) as extract:
            resp = client.post(
                "/api/chat/upload",
                data={
                    "session_id": session_id,
                    "file": (io.BytesIO(b"%PDF-1.4\n%%EOF\n"), "paper.pdf"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["n_pages"], 1)
        self.assertEqual(body["n_chunks"], 1)
        self.assertIn("paper.pdf", _session_docs[session_id]["doc_names"])
        self.assertEqual(_session_docs[session_id]["chunks"][0]["doc_name"], "paper.pdf")
        extract.assert_called_once()
        _session_docs.pop(session_id, None)

    def test_remove_session_doc_drops_only_selected_chunks(self):
        session_id = "remove_doc_regression"
        _session_docs[session_id] = {
            "doc_names": ["a.pdf", "b.pdf"],
            "chunks": [
                {"doc_name": "a.pdf", "page": 1, "text": "aaa"},
                {"doc_name": "b.pdf", "page": 1, "text": "bbb"},
            ],
        }
        client = web_app.app.test_client()

        resp = client.post(
            "/api/chat/remove_session_doc",
            json={"session_id": session_id, "doc_name": "a.pdf"},
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["doc_names"], ["b.pdf"])
        self.assertEqual(_session_docs[session_id]["chunks"], [
            {"doc_name": "b.pdf", "page": 1, "text": "bbb"},
        ])
        _session_docs.pop(session_id, None)

    def test_paper_read_uses_uploaded_pdf_not_web_or_skills(self):
        session_id = "paper_read_regression"
        _session_docs[session_id] = {
            "doc_names": ["seismicxmfinal.pdf"],
            "chunks": [
                {"doc_name": "seismicxmfinal.pdf", "page": 1, "text": "SeismicX is a seismology AI system with RAG, coding, and workflow agents."},
                {"doc_name": "seismicxmfinal.pdf", "page": 2, "text": "The method integrates knowledge retrieval with seismic data processing."},
            ],
        }

        with patch("routes.chat.get_llm_config", return_value={"api_base": "http://llm", "model": "test"}), \
             patch("routes.chat._chat_web_search_context", return_value=("UNRELATED WEB", [{"label": "web", "url": "https://example.test"}])) as web_search, \
             patch("routes.chat._skill_context_with_sources", return_value=("UNRELATED SKILL", "", [{"label": "skill", "url": ""}])) as skill_search, \
             patch("routes.chat.get_kb_instance", return_value=None):
            messages, sources, _ = chat_routes._build_rag_messages({
                "message": "解读一下这个文献。",
                "session_id": session_id,
                "mode": "paper_read",
                "enable_web_search": True,
            })

        system = messages[0]["content"]
        self.assertIn("当前上传文献", system)
        self.assertIn("SeismicX is a seismology AI system", system)
        self.assertNotIn("UNRELATED WEB", system)
        self.assertNotIn("UNRELATED SKILL", system)
        self.assertEqual(sources, [{"label": "[Uploaded PDF] seismicxmfinal.pdf", "url": "", "kind": "upload"}])
        web_search.assert_not_called()
        skill_search.assert_not_called()
        _session_docs.pop(session_id, None)

    def test_paper_read_without_uploaded_pdf_does_not_search_web(self):
        session_id = "paper_read_missing_doc"
        _session_docs.pop(session_id, None)

        with patch("routes.chat.get_llm_config", return_value={"api_base": "http://llm", "model": "test"}), \
             patch("routes.chat._chat_web_search_context", return_value=("UNRELATED WEB", [])) as web_search, \
             patch("routes.chat._skill_context_with_sources", return_value=("UNRELATED SKILL", "", [])) as skill_search, \
             patch("routes.chat.get_kb_instance", return_value=None):
            messages, sources, _ = chat_routes._build_rag_messages({
                "message": "解读一下这个文献。",
                "session_id": session_id,
                "mode": "paper_read",
                "enable_web_search": True,
            })

        system = messages[0]["content"]
        self.assertIn("NO CURRENT UPLOADED PAPER CONTENT IS AVAILABLE", system)
        self.assertNotIn("UNRELATED WEB", system)
        self.assertEqual(sources, [])
        web_search.assert_not_called()
        skill_search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
