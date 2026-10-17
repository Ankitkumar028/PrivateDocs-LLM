"""Optional Locust load test file for PrivateDocs-LLM."""

import os

from locust import HttpUser, between, task


class PrivateDocsUser(HttpUser):
    wait_time = between(0.1, 1.0)

    def on_start(self):
        self.api_key = os.getenv("USER_API_KEY", "user-secret-dev-key-change-in-prod")
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    @task(3)
    def ask_factual_question(self):
        self.client.post(
            "/v1/ask",
            headers=self.headers,
            json={"question": "What is the company operating revenue?", "top_k": 3},
            name="/v1/ask [factual]",
        )

    @task(1)
    def check_health(self):
        self.client.get("/healthz", name="/healthz")
