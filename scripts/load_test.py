# scripts/load_test.py
"""
Simulates concurrent users hitting the RAG inference server.
Run with: locust -f scripts/load_test.py --headless -u 20 -r 5 -t 60s --host http://localhost:8000
"""
from locust import HttpUser, task, between
import random

QUESTIONS = [
    "How do I return an item?",
    "When will I receive my refund?",
    "How do I track my order?",
    "What payment methods are accepted?",
    "My order is delayed, what should I do?",
    "What is Shopee Guarantee?",
    "How do I use a voucher?",
    "Can I change my delivery address?",
    "How do Shopee Coins work?",
    "What if seller doesn't respond?",
    # 👇 these are "unseen" queries that will always be cache misses
    "How do I cancel my order?",
    "Is cash on delivery available in my area?",
    "Why was my payment declined?",
    "How do I leave a review?",
    "Can I buy in bulk from one seller?",
]

METHODS = ["baseline", "reranker", "hyde"]

class RAGUser(HttpUser):
    # each user waits 0.5–2s between requests (realistic think time)
    wait_time = between(0.5, 2)

    @task(6)   # 60% — known questions (likely cache hits)
    def query_known(self):
        self.client.post(
            "/query",
            json={
                "question": random.choice(QUESTIONS[:10]),
                "method":   "baseline",
            },
            name="/query [baseline-known]",
        )

    @task(2)   # 20% — unseen questions (always cache miss)
    def query_unseen(self):
        self.client.post(
            "/query",
            json={
                "question": random.choice(QUESTIONS[10:]),
                "method":   "baseline",
            },
            name="/query [baseline-unseen]",
        )

    @task(2)   # 20% — reranker
    def query_reranker(self):
        self.client.post(
            "/query",
            json={
                "question": random.choice(QUESTIONS[:10]),
                "method":   "reranker",
            },
            name="/query [reranker]",
        )