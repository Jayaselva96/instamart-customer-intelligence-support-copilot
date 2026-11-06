# Instamart Customer Intelligence & Support Copilot

> **AI/ML Capstone Project**  
> An end-to-end customer analytics and support automation prototype built with Python, Pandas, NumPy, scikit-learn, SQLite, TF-IDF retrieval and a deterministic mock LLM.

---

## Project Overview

The **Instamart Customer Intelligence & Support Copilot** is a four-part analytics and AI/ML project. It transforms raw customer, product, order and support-ticket data into cleaned datasets, customer segments, churn/lapse predictions, support-ticket classifications and a policy-grounded refund-support agent.

The project is organized as a reproducible local prototype. It uses supplied/synthetic Instamart-style datasets and a deterministic mock LLM. It does **not** connect to live Instamart systems, make real refunds or use a production language-model API.

### Project at a Glance

| Area | Implementation |
|---|---|
| Part A | Raw data audit, data cleaning, EDA, SQLite database and SQL analytics |
| Part B | RFM analysis, K-Means customer segmentation, PCA and lapse prediction |
| Part C | Support-ticket triage, PII redaction, classification, priority routing and token-budget demonstration |
| Part D | Policy document chunking, TF-IDF retrieval, mock-LLM policy Q&A and simulated ReAct refund agent |
| Validation | Four part-specific validators, automated pytest checks and Python compilation |

## Table of Contents

1. Business Problem and Objectives
2. Dataset Overview
3. Technology Stack
4. Repository Structure
5. Part A — Data Audit, Cleaning and SQL Analytics
6. Part B — Customer Segmentation and Lapse Prediction
7. Part C — Support Ticket Intelligence
8. Part D — Policy RAG and Refund Support Agent
9. Installation and Environment Setup
10. How to Run the Project
11. Final Verification and Automated Tests
12. Generated Outputs
13. Limitations and Responsible Use

---

## 1. Business Problem and Objectives

An online grocery and quick-commerce business needs to understand customer purchasing behaviour, identify customers who may stop ordering, respond to customer support tickets efficiently, and provide consistent answers based on support policies.

This project demonstrates a local prototype for those needs.

### Main Objectives

- Inspect raw data and identify missing, duplicated and invalid values.
- Clean customer, product, order and ticket datasets.
- Store cleaned business data in a relational SQLite database.
- Answer business questions using SQL aggregations and window functions.
- Calculate Recency, Frequency and Monetary (RFM) features.
- Segment customers using K-Means clustering.
- Train and evaluate supervised models for customer lapse prediction.
- Classify support tickets into billing, delivery, technical and fraud categories.
- Redact common personal identifiers before ticket text is passed to the mock model.
- Retrieve relevant policy text and generate context-grounded support answers.
- Demonstrate refund decisions, human escalation, checkpoint persistence and recovery.

---

## 2. Dataset Overview

The project data is organized under `data/raw/`, `data/clean/` and `data/policies/`.

### Raw Datasets

| File | Rows | Purpose |
|---|---:|---|
| `customers.csv` | 1,200 | Customer profile, city, signup date, age and gender |
| `products.csv` | 40 | Product catalogue, category and price |
| `orders.csv` | 13,084 | Customer orders, quantities, timestamps, status, discounts and delivery time |
| `tickets.csv` | 164 | Customer support-ticket text and supplied category labels |
| `policy_eval.csv` | 14 | Questions and expected policy document/fact for retrieval evaluation |
| `agent_scenarios.csv` | 7 | Refund-agent test scenarios and expected outcomes |

### Policy Documents

The support assistant reads five text policy files from `data/policies/`:

- `refund_policy.txt`
- `delivery_policy.txt`
- `payment_policy.txt`
- `privacy_policy.txt`
- `escalation_policy.txt`

A JSON policy file, `support_policies.json`, is also included in the policy directory.

### Data Note

The project dataset is a supplied/synthetic demonstration dataset. The reported metrics describe this dataset and its evaluation setup; they should not be interpreted as validated business performance on real Instamart customer data.

---

## 3. Technology Stack

| Technology | Role in This Project |
|---|---|
| Python | Main programming language |
| Pandas | Data loading, cleaning, transformation and analysis |
| NumPy | Numerical operations and missing-value handling |
| scikit-learn 1.8.0 | Scaling, K-Means, PCA, model selection, classification and TF-IDF |
| SQLite | Local relational database |
| SQL | Business analytics, aggregations and window functions |
| Joblib | Saving and loading the selected lapse-prediction model |
| pytest | Automated project tests |
| JSON / JSONL / CSV | Configuration, checkpoints, ledgers and generated outputs |

### Python Dependencies

The current `requirements.txt` specifies:

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn==1.8.0
joblib>=1.3.0
pytest>=7.0.0
```

The project pins scikit-learn to **1.8.0**. Keep the saved model and the environment used to load it on the same scikit-learn version. If the model is retrained under a different version, update and test the dependency specification accordingly.

---

## 4. Repository Structure

```text
Instamart capstone project/
│
├── data/
│   ├── raw/
│   ├── clean/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   └── orders.csv
│   ├── policies/
│   └── instamart.db
│
├── notebooks/
│   ├── 01_Data_Audit_Cleaning_and_EDA.ipynb
│   ├── 02_SQL_Business_Analytics.ipynb
│   ├── 03_RFM_Segmentation_and_Churn_Prediction.ipynb
│   ├── 04_Support_Ticket_Triage_and_PII_Guardrails.ipynb
│   └── 05_Policy_RAG_and_Support_Copilot.ipynb
│
├── outputs/
│   ├── data_quality_report.json
│   ├── customer_segments.csv
│   ├── lapse_model.joblib
│   ├── retention_list.csv
│   ├── churn_predictions.csv
│   ├── churn_metrics.json
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   ├── agent_runs.jsonl
│   └── refund_ledger.jsonl
│
├── sql/
│   └── queries.sql
│
├── src/
│   ├── audit.py
│   ├── clean.py
│   ├── eda_part_a.py
│   ├── run_sql.py
│   ├── rfm_segmentation.py
│   ├── lapse_model.py
│   ├── ticket_triage.py
│   ├── policy_rag.py
│   ├── refund_agent.py
│   ├── validate_part_a.py
│   ├── validate_part_b.py
│   ├── validate_part_c.py
│   └── validate_part_d.py
│
├── state/
│
├── tests/
│   └── test_project.py
│
├── generate_data.py
├── mock_llm.py
├── run_project.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. Part A — Data Audit, Cleaning and SQL Analytics

### 5.1 Raw Data Audit

The raw datasets are inspected before cleaning to understand their dimensions, data types, missing values, duplicate records and invalid values.

The audit includes:

- Dataset row and column counts.
- Missing-value counts and percentages.
- Duplicate order rows.
- Distinct order IDs.
- Invalid or non-numeric order amounts.
- Negative and blank delivery-time values.
- Missing and out-of-range customer ages.
- Variations in city and gender values.

### 5.2 Raw Order Audit Checkpoints

The supplied raw order data contains the following observed values:

| Audit Metric | Observed Value |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending orders | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery-time values | 106 |
| Blank delivery-time values | 2,541 |
| Missing customer ages | 61 |
| Customer ages outside the 10–100 range | 6 |

### 5.3 Data Cleaning

**Customer cleaning**

- Standardizes city text by trimming spaces and applying consistent title case.
- Normalizes gender values into `male`, `female` and `unknown`.
- Converts age to numeric values.
- Treats ages outside the valid 10–100 range as missing.
- Imputes missing ages using the median age for the corresponding city.
- Standardizes signup dates.

**Product cleaning**

- Retains the product catalogue and its product ID, product name, category and INR price fields for downstream joins and order calculations.

**Order cleaning**

- Converts order timestamps into a consistent datetime representation.
- Removes exact duplicate rows.
- Retains the latest record for each order ID.
- Converts order amounts to numeric values.
- Recalculates missing order amounts using product price, quantity and discount:

```python
order_amount = price_inr * quantity * (1 - discount_pct / 100)
```

- Converts negative delivery times to missing values.
- Imputes missing delivery times for completed orders using the median for the relevant product category.
- Leaves cancelled-order delivery times missing.

**Support-ticket cleaning**

- Creates a `clean_text` field.
- Redacts email addresses, phone numbers and order-ID patterns from ticket text.

### 5.4 Cleaned Data Results

| Dataset | Cleaned Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Support tickets | 164 |

Additional checkpoints:

- Duplicate order records are removed.
- Cleaned order IDs are unique.
- Completed-order revenue: **INR 2,998,668.80**.

The results are written to `data/clean/` and summarized in `outputs/data_quality_report.json`.

### 5.5 SQLite Database

The main data-cleaning pipeline in `src/clean.py` creates and populates the local SQLite database at:

`data/instamart.db`

The database stores cleaned customer, product and order data for downstream SQL analytics.

The database creation and population are handled by the cleaning pipeline, and the SQL analytics are executed using `src/run_sql.py`.

### 5.6 SQL Analysis

Seven SQL queries are implemented in `sql/queries.sql` and executed by `src/run_sql.py`.

| Query | Business Question / SQL Concept |
|---|---|
| Q1 | Customers with completed-order spending above the specified threshold |
| Q2 | Revenue grouped by city and product category |
| Q3 | Products priced above the average catalogue price |
| Q4 | Customers with zero completed orders |
| Q5 | Customers whose total spending is above the average customer spend |
| Q6 | Customer spending rankings within each city using window functions |
| Q7 | Daily completed revenue and cumulative running revenue |

The SQL work demonstrates filtering, joins, grouping, aggregate functions, common table expressions and window functions such as `RANK()`, `ROW_NUMBER()` and cumulative sums.

The SQL runner includes reference-checkpoint assertions for the expected query results.

---

## 6. Part B — Customer Segmentation and Lapse Prediction

Part B contains two connected machine-learning tasks: unsupervised customer segmentation and supervised customer lapse prediction.

### 6.1 RFM Feature Engineering

RFM features are calculated from completed orders:

- **Recency:** Number of days since the customer's most recent completed order.
- **Frequency:** Number of unique completed orders.
- **Monetary:** Total amount spent on completed orders.

The segmentation script uses a snapshot date of **1 October 2025**.

### 6.2 K-Means Customer Segmentation

The RFM features are standardized using `StandardScaler`.

K-Means clustering is evaluated for values of K from 2 through 8. Inertia and silhouette score are used to inspect the candidate cluster configurations. The implementation selects **K = 4** for the final segmentation.

PCA with two components is also applied to the standardized RFM feature matrix for dimensionality-reduction analysis.

### 6.3 Customer Segment Profiles

The generated `outputs/customer_segments.csv` contains 1,175 customers with completed-order history.

| Customer Segment | Customers | Average Recency (Days) | Average Frequency | Average Monetary Value (INR) |
|---|---:|---:|---:|---:|
| Daily Grocery Loyalists | 419 | 8.77 | 19.40 | 3,410.02 |
| High-Value Premium Shoppers | 138 | 24.12 | 6.77 | 7,129.70 |
| Lapsed / At-Risk Occasionals | 182 | 109.04 | 2.20 | 685.99 |
| Steady Moderate Shoppers | 436 | 28.86 | 3.77 | 1,057.62 |
| **Total** | **1,175** | — | — | — |

**Segment interpretation and possible business actions**

- **Daily Grocery Loyalists:** Frequent recent shoppers; suitable for replenishment reminders, subscription benefits and loyalty offers.
- **High-Value Premium Shoppers:** Highest average monetary value; suitable for premium service and retention attention.
- **Lapsed / At-Risk Occasionals:** Longer time since last order and lower purchase frequency; suitable for win-back campaigns.
- **Steady Moderate Shoppers:** Moderate activity; suitable for relevant bundles and cross-category recommendations.

These names are descriptive interpretations of the clusters, not predefined customer labels.

### 6.4 Customer Lapse Prediction

The supervised lapse-prediction workflow:

1. Builds a customer cohort and features from the cleaned customer, product and order data.
2. Creates a train/test split.
3. Establishes an always-not-lapsed baseline.
4. Trains Logistic Regression and Random Forest classifiers.
5. Uses GridSearchCV with F1 scoring for hyperparameter selection.
6. Compares model precision, recall, F1 score and confusion matrices.
7. Runs a class-balance experiment using resampling.
8. Examines feature importance and Logistic Regression coefficients.
9. Saves the selected Random Forest model using Joblib.
10. Reloads the saved model and checks that predictions are reproduced.
11. Creates a top-10 retention list from premium customers in the test set.

The lapse-model script uses a cohort of 1,156 customers, with 924 training rows and 232 test rows in the recorded execution.

### 6.5 Churn Prediction Model Evaluation

The churn prediction model was evaluated using a train-test split.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 73.71% |
| Precision | 65.91% |
| Recall | 65.17% |
| F1-score | 65.54% |

**Dataset split:**

- Training samples: 924
- Testing samples: 232

The evaluation indicates that the model identifies customer churn patterns with a test accuracy of 73.71%. Precision, recall and F1-score provide additional insight into its ability to identify customers at risk of churn.

These results are based on the supplied project dataset and should not be interpreted as production performance.

### 6.6 Model Outputs

| File | Description |
|---|---|
| `outputs/customer_segments.csv` | RFM features, cluster ID, segment name and premium flag |
| `outputs/lapse_model.joblib` | Saved selected Random Forest model |
| `outputs/retention_list.csv` | Up to 10 premium test customers ranked by lapse probability |
| `outputs/churn_predictions.csv` | Prediction output from the separate churn-prediction artifact |
| `outputs/churn_metrics.json` | Metrics associated with that churn-prediction artifact |

The RFM/lapse workflow and the separate `prediction churn/` artifact are both retained in the project; their outputs should not be treated as the same evaluation run.

---

## 7. Part C — Support Ticket Intelligence

Part C builds a support-ticket triage pipeline using the supplied 164-ticket dataset and the local deterministic mock LLM.

### 7.1 Input Guardrails and Privacy

Before classification, the pipeline applies input checks and PII redaction.

- Empty or whitespace-only ticket text is skipped.
- Extremely long ticket input is flagged rather than passed through normal classification.
- Email addresses, phone numbers and order-ID patterns are redacted.
- Token estimates are used for input-budget demonstrations.

### 7.2 Ticket Classification

The allowed categories are:

- `billing`
- `delivery`
- `technical`
- `fraud`

The implementation compares zero-shot and few-shot prompt configurations.

Model responses are processed through a validation sequence:

1. Attempt to parse the response as JSON.
2. Check whether the returned category is one of the allowed categories.
3. If JSON validation fails, attempt a controlled keyword fallback.
4. If the output remains invalid or ambiguous, flag it for review instead of inventing a category.

### 7.3 Reliability and Routing

Transient mock-LLM failures are retried up to three times with exponential backoff and jitter.

Ticket routing uses the predicted category and customer segment to determine a queue and priority. The available routes include:

- `support_queue`
- `security_team`
- `premium_desk`
- `human_review`

Fraud tickets are routed to the security team with high priority. Premium customers can be routed to the premium desk. Invalid, empty or long inputs are handled through review/guardrail statuses.

### 7.4 Recorded Triage Results

The archived `outputs/triage_results.csv` contains 164 ticket rows.

| Triage Status | Tickets |
|---|---:|
| `classified_json` | 135 |
| `classified_keyword` | 12 |
| `flagged_invalid_output` | 9 |
| `flagged_long` | 4 |
| `skipped_empty` | 4 |
| **Total** | **164** |

**Recorded routing breakdown:**

| Route | Tickets |
|---|---:|
| `support_queue` | 102 |
| `security_team` | 34 |
| `human_review` | 17 |
| `premium_desk` | 11 |
| **Total** | **164** |

The output also records the ticket category, associated customer segment, priority and token usage.

### 7.5 Token Budget Demonstration

`outputs/triage_budget_demo.csv` stores the budget-limited demonstration. It records accepted classifications, keyword fallback, invalid outputs, long/empty input handling and budget-exhausted cases.

The budget demonstration is separate from the production triage output. Its results should be interpreted with the configured budget and the accepted/flagged status of each ticket.

### 7.6 Part C Output Files

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

---

## 8. Part D — Policy RAG and Refund Support Agent

Part D contains a policy-grounded question-answering component and a simulated tool-using refund agent.

### 8.1 Policy Document Chunking

The policy assistant reads five text policy documents from `data/policies/`.

The implementation uses:

- Chunk size: **40 words**
- Word overlap: **10 words**
- Sliding-window chunking

Overlap helps retain context across chunk boundaries.

### 8.2 TF-IDF Retrieval

The policy chunks are converted into TF-IDF vectors using `TfidfVectorizer`.

For a question:

1. The query is vectorized using the fitted TF-IDF vocabulary.
2. Cosine similarity is calculated against the policy chunks.
3. The top three chunks are retrieved.
4. The retrieved policy text and source filenames are passed as context to the mock LLM.
5. The assistant is instructed to answer only from the supplied context and to use a fallback when the context does not contain the answer.

### 8.3 Policy Retrieval Evaluation

The supplied `data/raw/policy_eval.csv` contains **14 evaluation questions**:

- 12 answerable questions with an expected policy document and expected fact.
- 2 unanswerable questions intended to test the fallback response.

**Recorded evaluation results:**

| Evaluation Metric | Standard Chunking | Zero-Overlap Chunking |
|---|---:|---:|
| Expected document match | 0/12 | 0/12 |
| Expected fact match | 12/12 | 10/12 |
| Unanswerable fallback match | 2/2 | 2/2 |

The standard RAG implementation successfully matched the expected facts and fallback responses, but did not retrieve the expected source document for the 12 answerable questions.

This indicates a limitation in document-level retrieval accuracy. The evaluation results should not be interpreted as perfect policy retrieval performance.

The policy evaluation function measures expected-document retrieval, expected-fact inclusion and unanswerable-question fallback behaviour. It also includes a zero-overlap chunking experiment to compare retrieval behaviour when chunk overlap is removed.

### 8.4 Simulated Refund Agent

The refund agent follows a ReAct-style loop: it receives a request, calls tools, observes tool results and continues until it produces a final outcome or reaches a safety limit.

The available tools include:

| Tool | Purpose |
|---|---|
| `get_order` | Look up the order and its details |
| `search_policy` | Retrieve relevant policy information |
| `calculate_refund` | Calculate a simulated refund amount |
| `issue_refund` | Record a simulated refund action |
| `escalate_to_human` | Record escalation for human review |

The refund action is simulated and logged locally. It does not transfer money or access a live payment gateway.

### 8.5 Seven-Scenario Agent Evaluation

The Part D validator checks seven scenario IDs and their expected outcomes.

| Scenario | Expected Status |
|---|---|
| S1 | `REFUND` |
| S2 | `REFUND` |
| S3 | `ESCALATE` |
| S4 | `DENIED` |
| S5 | `DENIED` |
| S6 | `NOT_FOUND` |
| S7 | `FALLBACK_MAX_ITERATIONS` |

The validator also checks that S1 uses five model calls and that the refund ledger contains both refund-issuance and human-escalation actions.

### 8.6 Checkpoint and Crash Recovery

The agent saves its current conversation, token usage, step number, executed tools and tool observations into JSON checkpoint files under `state/`.

The crash-recovery demonstration:

1. Starts scenario S1.
2. Simulates a crash after the second tool observation.
3. Preserves the checkpoint.
4. Loads the checkpoint and resumes the agent.
5. Continues with the remaining tools without repeating the completed tool actions.

The expected tool sequence across the complete resumed run is:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

### 8.7 Token Limit and Timeout Safety

The agent includes safety circuit breakers for:

- Token-limit exhaustion.
- Execution timeout.
- Maximum iteration exhaustion.

When a limit is reached, the agent returns a fallback outcome rather than continuing indefinitely.

### 8.8 Part D Output Files

| File | Purpose |
|---|---|
| `outputs/agent_runs.jsonl` | Recorded scenario run results |
| `outputs/refund_ledger.jsonl` | Simulated refund and escalation actions |
| `state/*.json` | Agent checkpoints and recovery state |

---

## 9. Installation and Environment Setup

### Prerequisites

- Windows 10/11 or another Python-supported operating system.
- Python installed and available from the terminal.
- Visual Studio Code (recommended).
- Git (for GitHub submission).

### Step 1 — Open the Project Folder

Open the project root folder in VS Code.

All terminal commands below must be run from the directory containing `run_project.py` and `requirements.txt`.

### Step 2 — Create a Virtual Environment

```powershell
python -m venv .venv
```

### Step 3 — Activate the Environment in PowerShell

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should show `(.venv)` after successful activation.

### Step 4 — Install Dependencies

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r requirements.txt
```

### Step 5 — Confirm the scikit-learn Version

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

Expected version for this project environment:

```text
1.8.0
```

Use the same dependency environment when loading the saved `outputs/lapse_model.joblib` model.

---

## 10. How to Run the Project

The project provides a main execution script named `run_project.py`.

It executes the complete Part A–D workflow in the correct order, including the four part-specific validation scripts.

Run all commands from the project root directory.

### Step 1 — Execute the Complete Project

```powershell
python run_project.py
```

### Step 2 — Review Generated Outputs

After execution, review the generated files under:

- `data/clean/`
- `outputs/`
- `state/`

### Step 3 — Review Validation Results

The main execution workflow includes the four part-specific validation scripts. Review their execution results to confirm that the project components complete successfully.

---

## 11. Final Verification and Automated Tests

The project includes automated tests using pytest to validate important functionalities across data processing, ticket triage, policy retrieval and refund-agent operations.

### Recorded Test Results

| Test Result | Count |
|---|---:|
| Total tests | 14 |
| Passed | 14 |
| Failed | 0 |

**Recorded execution time:** 35.49 seconds.

### Run Automated Tests

```powershell
python -m pytest -v
```

### Additional Verification

The project also includes four part-specific validation scripts:

- `src/validate_part_a.py`
- `src/validate_part_b.py`
- `src/validate_part_c.py`
- `src/validate_part_d.py`

These validators check the outputs and expected behaviour of the corresponding project components.

---

## 12. Generated Outputs

The project generates the following outputs during execution.

| Output | Description |
|---|---|
| `outputs/data_quality_report.json` | Raw-data and cleaning summary |
| `outputs/customer_segments.csv` | Customer RFM features and cluster labels |
| `outputs/lapse_model.joblib` | Saved Random Forest lapse model |
| `outputs/retention_list.csv` | Highest lapse-probability premium test customers |
| `outputs/churn_predictions.csv` | Churn prediction records |
| `outputs/churn_metrics.json` | Churn evaluation metrics |
| `outputs/triage_results.csv` | Ticket classifications and routing |
| `outputs/triage_budget_demo.csv` | Token-budget demonstration results |
| `outputs/agent_runs.jsonl` | Refund-agent scenario runs |
| `outputs/refund_ledger.jsonl` | Simulated refund/escalation ledger |

The project also creates SQLite tables in `data/instamart.db` and JSON checkpoint files in `state/`.

---

## 13. Limitations and Responsible Use

- **Synthetic/supplied data:** Results are based on the project datasets and do not establish performance on live Instamart customer data.
- **RFM interpretation:** K-Means cluster labels are descriptive interpretations of cluster profiles. They are not independently verified business segments.
- **Lapse prediction:** Model metrics are based on the project's cohort, feature construction and holdout split. They should not be treated as production churn performance.
- **Ticket classification:** The mock LLM and supplied ticket labels provide a controlled demonstration, not proof of production-level classification reliability.
- **PII redaction:** The implemented patterns cover common email, phone-number and order-ID formats. They are not a complete guarantee that every sensitive value will be detected.
- **Mock LLM:** The assistant uses a deterministic local mock implementation. It is not connected to a hosted LLM API.
- **Policy retrieval:** TF-IDF and cosine similarity provide lexical retrieval. A relevant policy fact can still be missed or incompletely retrieved.
- **Refund agent:** Refund issuance and escalation are simulated local tool actions. No real customer account, payment system or refund gateway is connected.
- **Checkpoint recovery:** The recovery demonstration validates the implemented local checkpoint flow; it is not a distributed production workflow.

Human review is important for ambiguous ticket classifications, sensitive cases and any real-world refund decision.

---

## Project Summary

This capstone demonstrates an end-to-end customer intelligence and support workflow:

**Raw Data → Data Quality and Cleaning → SQLite and SQL Analytics → RFM Segmentation → Lapse Prediction → Ticket Triage → Policy Retrieval → Simulated Refund Agent → Validation**

The project combines data engineering, SQL, unsupervised learning, supervised machine learning, text processing, retrieval-based question answering and tool-using agent concepts in one reproducible local implementation.

---

**Project:** Instamart Customer Intelligence & Support Copilot  
**Project Type:** AI/ML Capstone Project  
**Primary Language:** Python  
**Execution:** Local and offline prototype  
