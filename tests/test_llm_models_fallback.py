"""Tests for online model-list fallback behavior."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError


PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import app as web_app


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class TestOnlineModelsFallback(unittest.TestCase):
    def test_models_400_falls_back_to_chat_probe_current_model(self):
        def fake_urlopen(req, timeout=0):
            url = req.full_url
            if url.endswith("/models"):
                raise HTTPError(url, 400, "Bad Request", hdrs=None, fp=io.BytesIO(b"bad"))
            if url.endswith("/chat/completions"):
                return _Resp({"choices": [{"message": {"content": "ok"}}]})
            raise AssertionError(url)

        client = web_app.app.test_client()
        with patch("routes.llm._ur.urlopen", side_effect=fake_urlopen):
            resp = client.get(
                "/api/llm/online/models",
                query_string={
                    "provider": "custom",
                    "api_base": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                    "api_key": "sk-test",
                    "model": "qwen3.6-plus",
                },
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["models"], ["qwen3.6-plus"])
        self.assertEqual(body["source"], "chat_probe")
        self.assertIn("warning", body)


if __name__ == "__main__":
    unittest.main()
