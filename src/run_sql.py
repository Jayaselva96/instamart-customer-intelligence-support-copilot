"""Execute the seven Instamart SQL queries and validate project checkpoints."""
from pathlib import Path
import sqlite3
import pandas as pd

# Resolve paths from the project root, even when launched from another folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
SQL_FILE = PROJECT_ROOT / "sql" / "queries.sql"

def run_queries():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")

    content = SQL_FILE.read_text(encoding="utf-8")
    # Each query in this project ends with a semicolon.
    query_blocks = [q.strip() for q in content.split(";") if q.strip()]
    if len(query_blocks) != 7:
        raise ValueError(f"Expected 7 SQL queries, found {len(query_blocks)}")

    results = {}
    with sqlite3.connect(DB_PATH) as conn:
        print("================ EXECUTING SQL QUERIES ================\n")
        for i, query in enumerate(query_blocks, start=1):
            df = pd.read_sql_query(query, conn)
            results[f"Q{i}"] = df
            print(f"--- Q{i} Results (Shape: {df.shape[0]} rows, {df.shape[1]} cols) ---")
            print(df.head(5).to_string(index=False))
            print()

    print("================ SQL CHECKPOINT VERIFICATION ================")
    q1 = results["Q1"]
    assert len(q1) == 30, f"Q1 expected 30 rows, got {len(q1)}"
    print("Q1 passed: 30 customers")

    q2 = results["Q2"]
    assert len(q2) == 30, f"Q2 expected 30 rows, got {len(q2)}"
    q2_sum = round(q2["revenue"].sum(), 2)
    assert abs(q2_sum - 2998668.80) < 0.01, f"Q2 revenue sum mismatch: {q2_sum}"
    beng = q2.loc[(q2["city"] == "Bengaluru") & (q2["category"] == "Electronics"), "revenue"]
    assert len(beng) == 1 and abs(float(beng.iloc[0]) - 233909.45) < 0.01, "Q2 Bengaluru/Electronics benchmark mismatch"
    print("Q2 passed: 30 rows and revenue benchmarks")

    q3 = results["Q3"]
    assert len(q3) == 11, f"Q3 expected 11 rows, got {len(q3)}"
    print("Q3 passed: 11 products")

    q4 = results["Q4"]
    assert len(q4) == 25, f"Q4 expected 25 rows, got {len(q4)}"
    print("Q4 passed: 25 customers")

    q5 = results["Q5"]
    assert len(q5) == 550, f"Q5 expected 550 rows, got {len(q5)}"
    assert "avg_benchmark" in q5.columns, "Q5 missing avg_benchmark column"
    q5_avg = float(q5["avg_benchmark"].iloc[0])
    assert abs(q5_avg - 2552.06) < 0.01, f"Q5 expected average 2552.06, got {q5_avg:.2f}"
    print(f"Q5 passed: 550 customers; average benchmark {q5_avg:.2f}")

    q6 = results["Q6"]
    assert len(q6) == 1175, f"Q6 expected 1175 rows, got {len(q6)}"
    top_per_city = q6[q6["rnk"] == 1].set_index("city")
    ref_tops = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10),
    }
    for city, (customer_id, spend) in ref_tops.items():
        assert city in top_per_city.index, f"Q6 missing city: {city}"
        row = top_per_city.loc[city]
        assert row["customer_id"] == customer_id and abs(float(row["total_spend"]) - spend) < 0.01, f"Q6 top customer mismatch for {city}"
    rank_diff = int((q6["rnk"] != q6["row_num"]).sum())
    assert rank_diff == 8, f"Q6 expected 8 rank differences, got {rank_diff}"
    print("Q6 passed: city rankings and 8 RANK/ROW_NUMBER differences")

    q7 = results["Q7"]
    assert len(q7) == 183, f"Q7 expected 183 rows, got {len(q7)}"
    last_revenue = float(q7["cumulative_revenue"].iloc[-1])
    assert abs(last_revenue - 2998668.80) < 0.01, f"Q7 cumulative revenue mismatch: {last_revenue:.2f}"
    print("Q7 passed: 183 days and cumulative revenue benchmark")

    print("\nALL SQL QUERIES AND CHECKPOINTS PASSED PERFECTLY!")
    return results

if __name__ == "__main__":
    run_queries()
