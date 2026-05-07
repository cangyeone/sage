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
        extract.assert_called_once()
        _session_docs.pop(session_id, None)


if __name__ == "__main__":
    unittest.main()
