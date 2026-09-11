# LimitIQ — Causal Credit Limit Decisioning System

**Live demo:** [Streamlit Dashboard](https://limitiq-rfuf5xbaxtve943quyrmr6.streamlit.app/) · **API:** [https://limitiq-api.onrender.com](https://limitiq-api.onrender.com)

> Most credit risk models answer "how risky is this person?" LimitIQ answers a harder, more useful question: **"If we specifically raise this person's credit limit, what changes for them — and is it worth it?"**

---

## 1. The problem

Card issuers (Citi, Mastercard partner banks, PayPal Credit, etc.) make thousands of credit-limit decisions every day. The standard approach scores a customer's general default risk and applies a threshold. But that isn't actually the decision being made — the real question is causal: **what is the effect of raising *this* customer's limit**, separate from how risky they are in general?

This distinction matters because:
- A generally "risky" customer might respond fine to more available credit (lower utilization stress).
- A generally "safe" customer might overspend once given more room.
- Comparing customers who historically got limit increases against those who didn't is misleading on its own — banks have always favored their safer customers for increases, so naive comparisons are biased by this **confounding**.

LimitIQ is built to answer the causal question directly, not just the risk question.

---

## 2. What it actually does (input → output)

**Input:** a customer's profile — income, credit score, credit limit, monthly spend, utilization rate, late payment history, debt-to-income, credit inquiries, employment status, account type.

**Output:**
1. **Baseline risk score** — general probability of default
2. **Causal effect on default** — the specific, isolated effect of a limit increase on that risk
3. **Predicted spend lift** — how much more the customer is expected to spend
4. **Expected annual benefit / cost / net value** — the decision expressed in real currency (₹), not an arbitrary score
5. **Recommendation** — approve / reduce / deny
6. **SHAP explanation** — which factors drove the risk score, for auditability
7. **Out-of-distribution warning** — flags when a customer's profile is unlike anything the model was trained on, so a human reviewer knows not to fully trust the number

Every decision is logged to an audit trail (customer, requester, scores, decision, timestamp) — mirroring real regulatory requirements for credit decisions (adverse-action documentation, model risk management).

---

## 3. Tech stack — and why each piece was chosen

| Component | Choice | Why |
|---|---|---|
| Risk model | **CatBoost** | Handles mixed numeric/categorical features natively without manual encoding, strong out-of-the-box performance on tabular data, and widely used in real credit risk teams. Benchmarked against LightGBM during development — CatBoost won on validation AUC. |
| Causal model | **EconML (T-Learner)** | EconML is Microsoft's causal ML library, purpose-built for exactly this problem: estimating heterogeneous treatment effects when treatment assignment isn't random. Benchmarked T-Learner vs X-Learner vs DR-Learner on Qini coefficient — T-Learner performed best on this dataset, so it's the one deployed. |
| Calibration | **Platt scaling (logistic regression)** | The risk model uses class-balanced training to handle the ~7% default base rate, which distorts raw probabilities. Platt scaling is the standard fix — a simple logistic regression that remaps distorted probabilities back to real-world frequencies. |
| Explainability | **SHAP** | Industry-standard for tree-based model explanations; required in lending because adverse-action notices must state *which factors* drove a decision, not just the decision itself. |
| API | **FastAPI** | Async, automatic request validation via Pydantic, and the de facto standard for serving ML models in production Python stacks. |
| Auth | **JWT (python-jose)** | Any real internal banking tool requires authentication before use; JWT is stateless and standard for API-based auth. |
| Audit database | **SQLite + SQLAlchemy** | Lightweight for a portfolio-scale deployment; SQLAlchemy makes it trivial to swap in Postgres for a real production system without changing application code. |
| Dashboard | **Streamlit** | Fastest way to build an interactive, usable interface for a non-technical reviewer (a credit analyst) without writing custom frontend code. |
| Containerization | **Docker** | Standard for reproducible deployment; same image runs identically locally and on Render. |
| Deployment | **Render (API) + Streamlit Community Cloud (dashboard)** | Free tiers sufficient for a portfolio demo; separates the stateless API from the interactive frontend, matching how these are typically deployed independently in production. |

---

## 4. Why two models instead of one

A single classifier can only answer "is this person risky." It cannot answer "what happens if we act." Two customers with an *identical* risk score can have opposite reactions to a limit increase — one uses the extra room responsibly, the other overspends. This is **heterogeneous treatment effect**, and it requires a causal model, not a classifier, because:

- The naive approach (compare average outcomes of treated vs. untreated customers) is biased — banks have always given increases to their safer customers, so treated customers *look* safer regardless of whether the increase itself helped.
- The T-Learner works around this by training **two separate risk models** — one on customers who got an increase, one on customers who didn't — then estimates each customer's effect as the difference between what both models would predict for them. This isolates the effect of the treatment from the effect of who historically got treated.

---

## 5. Why an expected-value decision rule, not a fixed threshold

An earlier version of this system used a hard percentage cutoff (e.g., "approve if causal risk increase < 3%"). This is arbitrary and doesn't reflect how banks actually think — a small risk increase might be worth it if the extra spending revenue outweighs it, and vice versa.

The current decision rule computes:
```
expected_benefit = (extra annual spend) × (profit margin on spend)
expected_cost    = (causal default risk increase) × (credit exposure) × (loss given default)
net_value        = expected_benefit − expected_cost
```
with a hard safety cap: regardless of the economics, a causal risk increase above a fixed threshold is an automatic deny. This mirrors how real credit risk teams frame limit decisions — in terms of expected monetary value, not an unexplained score cutoff.

---

## 6. Why out-of-distribution detection exists

During scenario testing, an extreme synthetic customer (utilization rate 1.3, five late payments, unemployed, four months on book) produced an unstable, non-monotonic risk score — the model was extrapolating into territory with **zero examples** in the 50,000-row training set. Rather than silently return an unreliable number, LimitIQ checks whether a customer's raw feature values fall outside the 1st–99th percentile range of training data and flags the prediction as low-confidence when they do. This is a known limitation of tree-based models and a real safeguard used in production risk systems — most portfolio projects never implement it.

---

## 7. Bugs found and fixed during development (and why they matter)

Documenting these honestly, because finding and fixing them is a stronger signal of understanding than a project that "just worked":

**a) Data generation collapsed all signal to noise.** The synthetic risk formula originally weighted raw income (scale ~₹9,00,000) far more heavily than every other feature combined, saturating the default probability to a floor value for nearly everyone. Result: ROC-AUC of 0.50 (random). Fixed by z-score standardizing every component before combining them.

**b) Calibration was badly distorted by class balancing.** Class-balanced training (needed because only ~7% of customers default) skewed raw probabilities — a 43% calibration gap. Fixed with Platt scaling, bringing the gap under 5%.

**c) The causal effect direction was backwards.** The synthetic "true" treatment effect initially made *lower*-risk customers show a *larger* risk increase from a limit hike — the opposite of real-world intuition. Caught via automated scenario testing (a risky customer's causal effect should exceed a safe customer's), and fixed by correcting the formula so the effect scales with debt-to-income in the right direction.

**d) Engineered interaction features caused false out-of-distribution alarms.** Adding multiplicative features (e.g., credit score × income) meant a genuinely safe, simply above-average customer produced an extreme *product* value even though every individual input was normal, tripping the OOD check. Fixed by restricting the OOD check to raw, meaningful features only — not derived interaction terms.

**e) A deployment-time `NameError`.** A variable was defined as `_training_df` but referenced as `training_df` in one function call, crashing the live API on every request. Found by reproducing the exact failure locally (where Python's traceback is far easier to read than a hosting platform's log viewer) rather than debugging blind against the deployed logs.

---

## 8. Validated results (current)

| Metric | Value | Interpretation |
|---|---|---|
| Risk model ROC-AUC | ~0.68 | Real, learnable signal — comparable to a reasonable (not top-tier) credit model, an honest result for synthetic data with deliberate noise |
| Calibration gap (after Platt scaling) | ~2–5% | Predicted probabilities closely match real-world frequencies |
| Uplift model Qini coefficient | ~0.01 | Positive — the model finds real, if modest, heterogeneous treatment effects; causal effects are inherently harder to validate than plain predictions since no customer's counterfactual outcome is ever observed |
| Scenario monotonicity (in-distribution) | Pass | Risk score and causal effect increase sensibly across safe → risky profiles |
| Out-of-distribution detection | Correctly flags extreme, unseen profiles without false-flagging normal high-income customers | |

A full automated validation suite (`tests/run_full_evaluation.py`) reproduces all of the above in one command.

---

## 9. Project structure

```
limitiq/
├── data/                    # simulated training data (raw + processed)
├── models/                  # trained model artifacts
├── src/
│   ├── data_simulation.py   # generates realistic synthetic credit data
│   ├── feature_engineering.py
│   ├── risk_model.py        # CatBoost default risk classifier
│   ├── uplift_model.py      # EconML T-Learner causal uplift model
│   ├── calibration.py       # Platt scaling
│   └── explain.py           # SHAP + out-of-distribution detection
├── api/                     # FastAPI backend (auth, decisioning, audit log)
├── dashboard/               # Streamlit frontend
└── tests/                   # scenario tests + full validation suite
```

---

## 10. Running it locally

```bash
python -m venv venv
venv\Scripts\activate        # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt

python -m src.data_simulation
python -m src.feature_engineering
python -m src.risk_model
python -m src.uplift_model
python -m src.calibration

uvicorn api.main:app --reload
# in a second terminal:
streamlit run dashboard/app.py
```

Run the validation suite:
```bash
python -m tests.run_full_evaluation
python -m tests.test_scenarios
```

---

## 11. Known limitations

- **Synthetic data.** Real bank data isn't publicly available; the simulation is designed with intentional noise and realistic confounding, but a real deployment would need to validate against actual historical outcomes.
- **SQLite audit log is not persistent on free-tier hosting** — Render's free instance filesystem resets on redeploy. A production deployment would use a managed Postgres database.
- **Causal effects can't be validated the way classifiers can** — there's no ground-truth counterfactual for any individual customer, so causal model evaluation (Qini coefficient) is inherently a looser standard than classification accuracy.
- **Free-tier hosting** means the API may take 30–60 seconds to respond after periods of inactivity (cold start).