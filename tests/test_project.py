
from pathlib import Path
import sys
import json
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ticket_triage import redact_pii, parse_model_response
from src.policy_rag import chunk_documents, PolicyRetriever
from src import refund_agent


# -------------------------
# PART A / PART B TESTS
# -------------------------

def test_customer_files_exist():
    assert (ROOT / "data/raw/customers.csv").exists()
    assert (ROOT / "data/clean/customers.csv").exists()


def test_clean_customer_ids_unique():
    df = pd.read_csv(ROOT / "data/clean/customers.csv")
    assert "customer_id" in df.columns
    assert df["customer_id"].is_unique


def test_churn_is_binary():
    df = pd.read_csv(ROOT / "outputs/churn_predictions.csv")
    assert "churned" in df.columns
    assert set(df["churned"].dropna().unique()).issubset({0, 1})

# -------------------------
# PART C TESTS
# -------------------------

def test_pii_redaction():
    text = "Email john@gmail.com, phone 9876543210, order ORD-123456"

    redacted, emails, phones, orders = redact_pii(text)

    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted
    assert "[ORDER_ID]" in redacted
    assert emails == 1
    assert phones == 1
    assert orders == 1


def test_ticket_json_classification():
    response = '{"category": "billing", "reason": "Payment issue"}'

    status, category = parse_model_response(response)

    assert status == "classified_json"
    assert category == "billing"


def test_ticket_keyword_classification():
    response = "The customer has a delivery problem"

    status, category = parse_model_response(response)

    assert status == "classified_keyword"
    assert category == "delivery"


def test_invalid_ticket_classification():
    response = '{"category": "unknown"}'

    status, category = parse_model_response(response)

    assert status == "flagged_invalid_output"
    assert category is None


def test_triage_results_and_routing():
    path = ROOT / "outputs/triage_results.csv"

    assert path.exists()

    df = pd.read_csv(path)

    required_columns = {
        "ticket_id", "customer_id", "status",
        "category", "priority", "route"
    }

    assert required_columns.issubset(df.columns)

    valid_routes = {
        "human_review", "security_team",
        "premium_desk", "support_queue"
    }

    assert set(df["route"].dropna().unique()).issubset(valid_routes)


# -------------------------
# PART D: POLICY RAG TESTS
# -------------------------

def test_policy_documents_are_chunked():
    chunks = chunk_documents(overlap=10, chunk_size=40)

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk["source"].endswith(".txt")
        assert len(chunk["text"].split()) <= 40
        assert len(chunk["text"].strip()) > 0


def test_policy_retriever_returns_results():
    chunks = chunk_documents(overlap=10, chunk_size=40)
    retriever = PolicyRetriever(chunks)

    results = retriever.retrieve("refund policy late delivery", k=3)

    assert len(results) > 0
    assert len(results) <= 3

    for result in results:
        assert "source" in result
        assert "score" in result
        assert "text" in result
        assert 0 <= result["score"] <= 1


# -------------------------
# PART D: REFUND AGENT TESTS
# -------------------------

def test_refund_calculation():
    result = refund_agent.calculate_refund(500, 46)
    assert result["refund_inr"] == 500.0

    result = refund_agent.calculate_refund(500, 45)
    assert result["refund_inr"] == 0.0

    result = refund_agent.calculate_refund(500, None)
    assert result["refund_inr"] == 0.0


def test_get_order_from_database(tmp_path, monkeypatch):
    test_db = tmp_path / "test_orders.db"

    connection = sqlite3.connect(test_db)

    connection.execute("""
        CREATE TABLE orders (
            order_id TEXT,
            status TEXT,
            order_amount REAL,
            delivery_minutes REAL
        )
    """)

    connection.execute(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        ("ORD-123456", "completed", 500.0, 50.0)
    )

    connection.commit()
    connection.close()

    monkeypatch.setattr(refund_agent, "DB_PATH", str(test_db))

    result = refund_agent.get_order("ORD-123456")

    assert result["order_id"] == "ORD-123456"
    assert result["status"] == "completed"
    assert result["order_amount"] == 500.0
    assert result["delivery_minutes"] == 50.0

    missing = refund_agent.get_order("ORD-999999")
    assert "error" in missing


def test_refund_action_is_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        refund_agent, "OUTPUTS_DIR", str(tmp_path)
    )

    result = refund_agent.issue_refund("ORD-123456", 500)

    assert result["refund_inr"] == 500.0

    ledger = tmp_path / "refund_ledger.jsonl"

    assert ledger.exists()

    with open(ledger, "r", encoding="utf-8") as file:
        record = json.loads(file.readline())

    assert record["tool"] == "issue_refund"
    assert record["order_id"] == "ORD-123456"
    assert record["refund_inr"] == 500.0


def test_agent_max_iterations_fallback():
    result = refund_agent.run_agent(
        user_message="Please check my refund",
        scenario_id="TEST_MAX_ITERATIONS",
        max_iterations=0
    )

    assert result["status"] == "FALLBACK_MAX_ITERATIONS"
