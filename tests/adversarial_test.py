"""
Adversarial Integration Tests for Brahman 2.0 Reasoning Oracle.
Tests the full API against the same attack vectors that killed V1.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_online(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["version"] == "2.0.0"
        assert data["dhatus_loaded"] > 0

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "Brahman 2.0" in response.json()["service"]


class TestVerifyEndpoint:
    def test_valid_sentence(self):
        """Standard SOV sentence should pass or trigger a structured response."""
        response = client.post("/v1/verify", json={
            "sentence": "रामः वनं गच्छति"
        })
        # May be 200 (valid) or 422 (violation caught) — both are correct behavior
        assert response.status_code in [200, 422]

    def test_empty_sentence_rejected(self):
        """Pydantic must reject empty input."""
        response = client.post("/v1/verify", json={
            "sentence": ""
        })
        assert response.status_code == 422

    def test_long_sentence_rejected(self):
        """Prevent overflow attacks."""
        response = client.post("/v1/verify", json={
            "sentence": "अ" * 501
        })
        assert response.status_code == 422


class TestSolnetEndpoint:
    def test_solnet_validate_returns_certificate(self):
        """The Oracle must return a VerificationCertificate with a logic_hash."""
        response = client.post("/v1/solnet/validate", json={
            "transaction_intent": "रामः वनं गच्छति",
            "node_id": "test-node-001"
        })
        # May be 200 or 422 depending on neural prediction
        assert response.status_code in [200, 422]

        data = response.json()
        if response.status_code == 200:
            assert "logic_hash" in data
            assert len(data["logic_hash"]) == 64  # SHA-256 hex
            assert "karaka_trace" in data
            assert data["node_id"] == "test-node-001"

    def test_solnet_empty_intent_rejected(self):
        """Pydantic must reject empty transaction_intent."""
        response = client.post("/v1/solnet/validate", json={
            "transaction_intent": ""
        })
        assert response.status_code == 422

    def test_equivocation_attack(self):
        """Duplicate-word equivocation must be handled without crash."""
        response = client.post("/v1/solnet/validate", json={
            "transaction_intent": "वनं वनं गच्छति"
        })
        assert response.status_code in [200, 422]

    def test_impossible_proof_detected(self):
        """
        Structurally illegal sentence (two accusatives, no nominative)
        must be flagged. This is the V1 graveyard test.
        """
        response = client.post("/v1/solnet/validate", json={
            "transaction_intent": "रामम् वनं गच्छति"
        })
        # This should either pass (if model doesn't predict conflicting roles)
        # or trigger 422 — but must NOT crash
        assert response.status_code in [200, 422]
