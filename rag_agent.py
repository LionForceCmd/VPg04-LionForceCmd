"""Базовый RAG-агент: Pinecone + OpenAI embeddings + LangChain agent.

Возможности:
  * подключение к векторному хранилищу Pinecone (индекс создаётся при отсутствии);
  * add_url(): парсинг страницы -> чанки -> эмбеддинги -> запись в Pinecone;
  * search(): семантический поиск по базе;
  * ask(): агент сам решает, когда искать в базе, читать URL или звать API;
  * @tool create_api_request_tool: реальный GET-запрос к открытому API (Cat Facts).

Запуск как модуля (`python rag_agent.py`) выполняет тестовую проверку подключения.
"""
from __future__ import annotations

import os
import re
import sys
import uuid

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

CAT_FACT_API = "https://catfact.ninja/fact"
HTTP_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; RAGAgent/1.0)"
EMBEDDING_DIM = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


# --------------------------------------------------------------------------
# Tool: реальный GET-запрос к открытому API (вместо заглушки)
# --------------------------------------------------------------------------
def fetch_cat_fact(endpoint: str = CAT_FACT_API) -> str:
    """GET-запрос к Cat Facts API с обработкой ошибок. Всегда возвращает строку."""
    try:
        resp = requests.get(
            endpoint, headers={"Accept": "application/json"}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        return "Ошибка API: превышено время ожидания."
    except requests.HTTPError as e:
        return f"Ошибка API: HTTP {e.response.status_code}."
    except (requests.RequestException, ValueError) as e:
        return f"Ошибка API: {e}"
    fact = data.get("fact") if isinstance(data, dict) else None
    if not fact:
        return f"API вернул неожиданный ответ: {data}"
    return f"Случайный факт о кошках: {fact}"


@tool
def create_api_request_tool() -> str:
    """Получить случайный факт о кошках через открытый API (GET https://catfact.ninja/fact).

    Используй, когда пользователь просит факт о кошках / котах.
    """
    return fetch_cat_fact()


class RAGAgent:
    def __init__(self, index_name: str | None = None, namespace: str | None = None):
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not openai_key or not pinecone_key:
            raise RuntimeError("Задайте OPENAI_API_KEY и PINECONE_API_KEY в .env")

        base_url = os.getenv("OPENAI_BASE_URL") or None
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-agent")
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE", "default")
        emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        self.embeddings = OpenAIEmbeddings(
            model=emb_model, api_key=openai_key, base_url=base_url
        )
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            base_url=base_url,
            temperature=0,
        )

        # --- Pinecone ---
        self.pc = Pinecone(api_key=pinecone_key)
        self._ensure_index(EMBEDDING_DIM.get(emb_model, 1536))
        self.index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(
            index=self.index, embedding=self.embeddings, namespace=self.namespace
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        self.agent = create_agent(
            self.llm,
            tools=[self._make_search_tool(), self._make_add_url_tool(), create_api_request_tool],
            system_prompt=(
                "Ты полезный ассистент с долговременной памятью в векторной базе. "
                "Перед ответом на фактологический вопрос вызывай search_knowledge_base. "
                "Если пользователь дал ссылку и просит прочитать/запомнить страницу — "
                "вызови read_url. Для фактов о кошках используй create_api_request_tool. "
                "Если в базе ничего нет — честно скажи об этом."
            ),
        )

    # ---------------------------------------------------------------- index
    def _ensure_index(self, dim: int) -> None:
        existing = {i["name"] for i in self.pc.list_indexes()}
        if self.index_name in existing:
            return
        print(f"[pinecone] создаю индекс '{self.index_name}' (dim={dim})")
        self.pc.create_index(
            name=self.index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # ------------------------------------------------------------------ URL
    @staticmethod
    def fetch_page_text(url: str) -> tuple[str, str]:
        """Скачивает страницу и возвращает (title, очищенный текст)."""
        if not re.match(r"^https?://", url):
            raise ValueError(f"Некорректный URL: {url}")
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for t in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            t.decompose()
        title = soup.title.get_text(strip=True) if soup.title else url
        text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n", strip=True))
        return title, text

    def build_documents(self, url: str) -> list[Document]:
        """Парсинг страницы и разбиение на чанки (без обращения к Pinecone)."""
        title, text = self.fetch_page_text(url)
        if not text.strip():
            raise ValueError("На странице нет текста для индексации")
        return [
            Document(page_content=c, metadata={"source": url, "title": title, "chunk": i})
            for i, c in enumerate(self.splitter.split_text(text))
        ]

    def add_url(self, url: str) -> int:
        """URL -> чанки -> эмбеддинги -> Pinecone. Возвращает число чанков."""
        docs = self.build_documents(url)
        # детерминированные id: повторная индексация той же страницы перезаписывает чанки
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#{i}")) for i in range(len(docs))]
        self.vector_store.add_documents(docs, ids=ids)
        print(f"[add_url] {url}: {len(docs)} чанков сохранено в Pinecone")
        return len(docs)

    # --------------------------------------------------------------- search
    def search(self, query: str, k: int = 4) -> list[Document]:
        return self.vector_store.similarity_search(query, k=k)

    # ----------------------------------------------------------------- tools
    def _make_search_tool(self):
        @tool
        def search_knowledge_base(query: str) -> str:
            """Семантический поиск по векторной базе знаний (Pinecone)."""
            docs = self.search(query)
            if not docs:
                return "В базе знаний ничего не найдено."
            return "\n\n".join(
                f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs
            )

        return search_knowledge_base

    def _make_add_url_tool(self):
        @tool
        def read_url(url: str) -> str:
            """Прочитать веб-страницу по URL и сохранить её в векторную базу знаний."""
            try:
                n = self.add_url(url)
            except Exception as e:  # noqa: BLE001 — отдаём ошибку агенту текстом
                return f"Не удалось обработать URL: {e}"
            return f"Страница сохранена, чанков: {n}."

        return read_url

    # ------------------------------------------------------------------ chat
    def ask(self, question: str) -> str:
        result = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content


def main() -> int:
    """Тестовая проверка: подключение к Pinecone + поиск + вызов API-инструмента."""
    agent = RAGAgent()
    stats = agent.index.describe_index_stats()
    print(
        f"[ok] Pinecone подключён, индекс '{agent.index_name}', "
        f"векторов всего: {stats.total_vector_count}"
    )

    docs = agent.search("What is this knowledge base about?", k=2)
    print(f"[ok] тестовый поиск выполнен, найдено документов: {len(docs)}")
    for d in docs:
        print("   -", d.metadata.get("source"), "|", d.page_content[:80].replace("\n", " "))

    print("[api]", create_api_request_tool.invoke({}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
