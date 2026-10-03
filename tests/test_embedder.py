"""Unit tests for Gemini embedding request construction."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.core.embedder import _embed_many


def test_embed_many_sends_one_content_per_text():
    client = MagicMock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[1.0, 0.0]),
            SimpleNamespace(values=[0.0, 1.0]),
        ]
    )

    with patch("app.core.embedder._get_embed_client", return_value=client):
        embeddings = _embed_many(
            ["Quản lý môi trường ao", "Phòng bệnh cho tôm"],
            "RETRIEVAL_DOCUMENT",
        )

    contents = client.models.embed_content.call_args.kwargs["contents"]
    assert len(contents) == 2
    assert contents[0].parts[0].text == "Quản lý môi trường ao"
    assert contents[1].parts[0].text == "Phòng bệnh cho tôm"
    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
