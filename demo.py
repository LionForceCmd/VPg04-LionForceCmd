"""Демо: python demo.py <url> [вопрос]"""
import sys

from rag_agent import RAGAgent

if __name__ == "__main__":
    agent = RAGAgent()
    if len(sys.argv) > 1:
        print("Чанков сохранено:", agent.add_url(sys.argv[1]))
    question = " ".join(sys.argv[2:]) or "Расскажи случайный факт о кошках"
    print("Q:", question)
    print("A:", agent.ask(question))
