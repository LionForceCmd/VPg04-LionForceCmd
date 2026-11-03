"""Офлайн-проверки без ключей: парсинг HTML, чанкинг и обработка ответов API."""
from unittest.mock import Mock, patch

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_agent

HTML = "<html><head><title>T</title><script>x=1</script></head><body><p>%s</p></body></html>" % (
    "Земля — третья планета от Солнца. " * 100
)


def test_parse_and_chunk():
    resp = Mock(text=HTML)
    resp.raise_for_status = Mock()
    with patch.object(rag_agent.requests, "get", return_value=resp):
        title, text = rag_agent.RAGAgent.fetch_page_text("https://example.com")
    assert title == "T" and "x=1" not in text
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_text(text)
    assert len(chunks) > 1 and all(len(c) <= 1000 for c in chunks)


def test_api_ok_and_errors():
    ok = Mock()
    ok.raise_for_status = Mock()
    ok.json.return_value = {"fact": "Cats sleep a lot", "length": 16}
    with patch.object(rag_agent.requests, "get", return_value=ok):
        assert "Cats sleep a lot" in rag_agent.fetch_cat_fact()
    with patch.object(rag_agent.requests, "get", side_effect=requests.Timeout):
        assert "время ожидания" in rag_agent.fetch_cat_fact()
    ok.json.return_value = {"x": 1}
    with patch.object(rag_agent.requests, "get", return_value=ok):
        assert "неожиданный" in rag_agent.fetch_cat_fact()


if __name__ == "__main__":
    test_parse_and_chunk()
    test_api_ok_and_errors()
    print("offline tests OK")
