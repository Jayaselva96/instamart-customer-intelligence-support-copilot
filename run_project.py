
"""
Instamart Customer Intelligence & Support Copilot
Main project execution script.

Run from the project root:
    python run_project.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(description, script):
    print("\n" + "=" * 65, flush=True)
    print(f"STARTING: {description}", flush=True)
    print("=" * 65, flush=True)

    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        raise FileNotFoundError(f"Required file not found: {script_path}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True
    )

    print(f"\nSUCCESS: {description}", flush=True)


def main():
    print("\nINSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    print("Starting complete project execution...\n")

    steps = [
        # Part A: Data Audit, Cleaning and SQL Analytics
        ("Part A - Data Audit", "src/audit.py"),
        ("Part A - Data Cleaning", "src/clean.py"),
        ("Part A - Exploratory Data Analysis", "src/eda_part_a.py"),
        ("Part A - SQL Business Analytics", "src/run_sql.py"),
        ("Part A - Validation", "src/validate_part_a.py"),

        # Part B: Customer Segmentation and Churn Prediction
        ("Part B - RFM Segmentation", "src/rfm_segmentation.py"),
        ("Part B - Churn Prediction", "src/lapse_model.py"),
        ("Part B - Validation", "src/validate_part_b.py"),

        # Part C: Support Ticket Triage and PII Guardrails
        ("Part C - Ticket Triage", "src/ticket_triage.py"),
        ("Part C - Validation", "src/validate_part_c.py"),

        # Part D: Policy RAG and Refund Support Agent
        ("Part D - Policy RAG Evaluation", "src/policy_rag.py"),
        ("Part D - Refund Agent Scenarios", "src/refund_agent.py"),
        ("Part D - Validation", "src/validate_part_d.py"),
    ]

    try:
        for description, script in steps:
            run_step(description, script)

    except subprocess.CalledProcessError as error:
        print("\nPROJECT EXECUTION FAILED.", flush=True)
        print(f"Failed script: {error.cmd}", flush=True)
        print(f"Exit code: {error.returncode}", flush=True)
        sys.exit(error.returncode)

    except Exception as error:
        print(f"\nPROJECT EXECUTION FAILED: {error}", flush=True)
        sys.exit(1)

    print("\n" + "=" * 65)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("ALL FOUR PART VALIDATIONS PASSED.")
    print("=" * 65)


if __name__ == "__main__":
    main()
