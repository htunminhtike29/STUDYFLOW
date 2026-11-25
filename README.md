# STUDYFLOW · Hybrid AI Smart Study Planner

**An Intelligent Study Scheduling System Using Machine Learning, Rule-Based Inference, and Heuristic Scheduling.**

University Artificial Intelligence final group project · Assessment weight 15%

| Group member | Student ID |
| --- | --- |
| Htun Min Htike | 6709684 |
| Min Mg Mg Kyaw | 6709687 |
| Aung Khant Zaw | 6709532 |
| Pyae Sone Soe | 6709586 |
| Min Thant | 6805378 |

## Abstract and objectives

Students must estimate the hours a task will take before they know its actual cost. An optimistic estimate can make a week appear feasible even when several deadlines compete for the same time. StudyFlow offers a learned second estimate, lets the student choose a study budget, and explains how that budget influences the schedule. Logging a session records what really happened. Confirming completion supplies a labeled example for future personal training.

The project demonstrates supervised regression for a continuous target, rule-based urgency inference, weighted ranking, priority-queue processing, constrained greedy allocation, and a private feedback loop. All training, inference, storage, and charts run locally. The release includes a trained model ready for a classroom demonstration.

## Windows quick start

Extract the full ZIP to a writable location and open PowerShell in the extracted `STUDYFLOW` directory. Use Python 3.11 or newer with compatible binary packages. The release was tested on 64-bit Windows with Python 3.15.0. To use an isolated environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit. Stop it with Ctrl+C. If PowerShell activation is unavailable, invoke `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m streamlit run app.py`. The pinned scikit-learn version matches the bundled Joblib artifact.

To reproduce training or regenerate a missing demonstration model:

```powershell
python train_model.py
```

The script reads the included labeled synthetic CSV, creates it deterministically if missing, writes the trained model and its JSON report, and prints the measured errors. Each training run receives a new version. Running the app requires no API keys or external database.

**Existing installation.** Stop the old app and retain a backup of its `studyflow.db`. Place the upgraded files beside that database, or copy the database into the extracted project before first launch. The release excludes the working student database. A new installation starts empty and registration generates three relative-deadline examples. An existing installation migrates on startup. Running the old app on an upgraded database is unsupported. A different local database path can be selected with the `STUDYFLOW_DB` environment variable.

## What changed from the original planner

| Original behavior | Hybrid release |
| --- | --- |
| A manually entered total drives scheduling | Students can preview a trained prediction and select the original estimate, the recommendation, or a custom budget. |
| Progress stops at the estimate and implies completion | Each session adds actual time independently of the budget. Completion requires a separate confirmation. |
| Aggregate progress only | Dated sessions, frozen creation inputs, actual completion histories, and account-specific model versions persist. |
| Four planner views | All four remain, with an additional Insights page, simulator, evaluation charts, and group credits in About StudyFlow. |
| Deadline rules and greedy scheduling | The same inference, scoring, tie-breaking and allocation logic operates on accepted remaining hours. |

The archive omits the working database. A fresh installation creates one on startup. To move an existing installation, stop the old app, keep a backup, and place a copy of its `studyflow.db` beside the upgraded `app.py` before starting it. The in-place working project uses its existing database automatically.

## Architecture and technology

| Module or artifact | Technical responsibility |
| --- | --- |
| `app.py` | Streamlit entry point, session and account gate, task editor, prediction preview, four planner views |
| `auth.py` | Registration, random salts, PBKDF2-HMAC-SHA256 with 600,000 iterations, constant-time digest comparison, logout |
| `database.py`, `migrations.py` | Parameterized SQLite repository, schema upgrades, task ownership, sessions, histories, private model transactions |
| `models.py` | Task snapshot, decimal duration handling, remaining planned workload |
| `ml_engine.py` | Trusted model loading, input validation, scikit-learn prediction, explanation contributions, deployment bounds and fallback |
| `train_model.py` | Deterministic synthetic data generation, Pandas cleaning, dataset split, model fit, evaluation, persistence and personal promotion |
| `ai_engine.py` | Explicit urgency rules, exact rational priority scores, deterministic ordering and explanations |
| `scheduler.py` | Seven daily min-heaps, decimal study blocks, subject diversity and deadline accounting |
| `insights.py` | Held-out evaluation, Plotly charts, prediction simulator, training controls, and team credits |
| `data/study_history.csv` | 400 labeled synthetic academic tasks with subject, creation features and actual-hours target |
| `ml_models/study_model.joblib` | Fitted scikit-learn estimator and matching evaluation metadata in one serving bundle |
| `ml_models/model_metadata.json` | Readable version, training time, coefficients, input ranges, deployment bound, split sizes, predictions and evaluation results |
| `assets/style.css`, `.streamlit/config.toml` | Responsive presentation, dark navy theme, cyan and violet accents |
| `tests/` | Original planner regression coverage and new migration, learning, persistence and end-to-end UI tests |

SQLite, datetime, heapq, hashlib, secrets, Decimal, Fraction and unittest come from Python. Streamlit, scikit-learn, NumPy, Pandas, Joblib and Plotly are installed from `requirements.txt`.

## Database and migration

Student IDs remain text, preserving leading zeros. The existing student records, password hashes, task IDs, ownership and deadline index remain in place. Migration uses a write transaction and a SQLite schema version. Repeated starts do not duplicate records.

- **tasks** retains `total_hours` as accepted budget. New columns retain original estimated hours, difficulty and deadline interval at creation, prediction and model version, budget source, accumulated actual hours, completion flag and timestamp. Later deadline, difficulty or budget edits preserve the creation snapshot. The original bounded `completed_hours` column remains for schema compatibility. New planner reads use `actual_hours`.
- **study_sessions** records student, task, local study date, duration and UTC insertion time. A write transaction inserts the session and updates cumulative actual time together.
- **study_history** admits at most one record per explicitly finished nonsample task. It snapshots creation features, original prediction if available, total actual hours, and completion time. At least one positive logged duration is required. Later edits preserve this historical snapshot.
- **personal_models** stores candidate versions, fitted Joblib bytes, activation status and evaluation metadata by student. A partial unique index permits one active model per account. Activation and deactivation share one transaction.
- **Legacy data** retains accumulated progress as the starting actual time. Previously finished tasks keep their completed status. The latest available estimate and difficulty are used to initialize legacy snapshots. Original creation values and prior session dates cannot be verified, so these tasks are excluded from personal training. Migration fabricates neither sessions nor completion timestamps.
- **Ownership** comes from the authenticated session. Composite student/task foreign keys enforce session and history ownership. Deleting a task also deletes its linked sessions and history. Sample reset uses sample keys, preserving user-created tasks even when their titles match. A trained private version persists until the account is removed.

## Supervised learning and data provenance

Multiple Linear Regression is appropriate for estimating a numerical duration. Scikit-learn's `LinearRegression.fit` learns an intercept and three slopes by minimizing squared residuals. Its `.predict` method supplies the live recommendation.

| Training input | Time of observation |
| --- | --- |
| Difficulty, integer 1–5 | Student creates the task |
| Days until deadline | Deadline minus local creation date |
| Original estimated hours | Before seeing or accepting a recommendation |

The target is **actual accumulated hours at explicit completion**. Task subject is descriptive. Accepted budgets, predictions, completed hours, completion timestamps and personal identities are excluded from the feature matrix. Completion time is used to order personal records for evaluation.

The demonstration CSV contains **synthetic data only**. A seed of 104 generates ten subjects, five difficulty levels, deadline intervals of 0–35 days, and estimated durations between 0.5 and 16 hours. The generator combines a linear signal with normally distributed noise whose spread increases with the estimate. It floors synthetic actual time at 0.1 h. See `synthetic_data()` for the coefficients, sampling procedure and rounding. Starter tasks from registration do not enter this dataset.

Numerical cleaning rejects invalid, non-finite and out-of-range features or invalid targets. The demo uses a fixed 80/20 random split. Only the 320 training rows enter the fit and determine the training ranges and prediction bound. The remaining 80 rows evaluate both the model and the original-estimate baseline.

### Equation and deployment policy

For the shipped model, the learned equation in hours, rounded here for readability, is:

```text
raw prediction = 0.64727 + 0.61327 × difficulty
                 − 0.03020 × days_until_deadline
                 + 0.98750 × estimated_hours
```

The simulator shows the raw output and the contribution of each input. A coefficient describes an association holding the other inputs fixed. A causal interpretation would require additional evidence.

Recommendations are clipped to a minimum of 0.1 h and a maximum equal to the ceiling of the largest training target, bounded at 1,000,000 h. The demo upper limit is 22 h. The result is rounded to the nearest tenth of an hour. Evaluation uses the same policy. The UI flags inputs outside the observed training range. A student can retain the original estimate or enter a custom budget.

### Measured demonstration performance

| Held-out metric | Linear Regression | Original-estimate baseline |
| --- | ---: | ---: |
| Mean absolute error, h | 1.79575 | 2.19500 |
| Mean squared error, h² | 4.89731 | 6.80209 |
| Root mean squared error, h | 2.21299 | 2.60808 |
| R² | 0.73518 | 0.63218 |

MAE describes a typical absolute miss in hours. MSE and RMSE weight larger errors more heavily. RMSE uses hours. R² measures the share of test-target variation explained relative to predicting the test mean. It can be negative and is undefined for constant targets. Both MAE and RMSE improved on this generated holdout. Real-world predictive performance remains unmeasured.

Insights draws actual-versus-predicted points, signed residuals (actual minus predicted), coefficient bars, a baseline comparison, and training/testing target distributions from the active version's saved outputs.

## Hybrid decision pipeline

1. The student enters the task, deadline, difficulty and original estimate. **Predict study hours** previews a versioned recommendation. Changing the task inputs invalidates that preview.
2. The student's selected hours become the budget. For an open task, remaining work equals `max(0, accepted_budget − actual_hours)`. Explicitly completed tasks have zero schedulable hours. An exhausted budget leaves the task open for more sessions or revision.
3. The original rules infer urgency from days until due. Overdue, today, and tomorrow score 5. Days 2–3 score 4, days 4–5 score 3, days 6–7 score 2, and later days score 1. Overdue work is reported separately.
4. For positive remaining work, normalized workload equals `5 × task_remaining / maximum_remaining`. Priority equals `0.50 × urgency + 0.30 × difficulty + 0.20 × normalized_workload`. Internal comparisons use full precision. The queue key is `(−score, deadline, −remaining_hours, task_id)`.
5. A seven-day greedy scheduler allocates up to half an hour per selection, with smaller fractional tails when needed. It honors daily capacity, remaining work and inclusive deadlines. It prefers a two-hour daily subject limit across case-insensitive names, relaxes it for necessary deadline work, and uses deferred candidates when spare capacity remains. Completed and overdue tasks receive no new blocks.
6. Weekly Plan accounts separately for scheduled hours, unmet hours due in the window, later balances, and overdue work. Each block shows the task's budget source. A changed limit, budget or session immediately changes the next plan calculation.
7. Recording study time and confirming completion update SQLite. Model parameters change only when training is explicitly run. The hybrid system combines a learned numerical input with transparent deterministic decisions about which work to schedule.

## Personal training and privacy

**Train Personal Model** becomes available after **30 eligible, explicitly completed personal tasks**. The button trains only from the signed-in account's history. The shared demonstration model is trained from the labeled synthetic file. The personal split uses the earliest 80% of completions for training and the latest 20% for testing, with task ID as the tie-breaker for identical timestamps.

The candidate becomes active only if **both MAE and RMSE are strictly lower** than that student's original-estimate baseline on the same held-out tasks. Rejected candidates retain their metrics for review and the currently serving model stays active. With insufficient records the demo remains available. When a personal artifact cannot be read, the app tries the demo model. With neither available, manual and custom budgets, rule-based ranking, scheduling, and progress continue to work.

## User guide and seven-part classroom demonstration

1. **Authentication.** Register a fresh leading-zero student ID at the default 3 h/day. Sign in and show the three generated starter subjects. Existing users can sign in after migration.
2. **Prediction.** In Subjects & Tasks add “Database design report”, due in three days, difficulty 4, original estimate 4 h. Select **Predict study hours**. The bundled model returns a raw estimate of 6.9598 h and recommends **7.0 h**. Expand the calculation.
3. **Hybrid ranking.** Select **ML recommendation** and add the task. Show the 4 h original estimate, 7 h accepted budget, and priority explanation. With untouched samples its initial priority is 4.20/5. The original estimate remains available in history even after budget edits.
4. **Scheduling.** Open Weekly Plan, inspect the daily allocations and budget-source captions. Expand **Why these tasks?**. Lower the daily limit to 1 h to demonstrate deadline shortfalls, then restore 3 h.
5. **Progress.** Select the report in Progress Tracker, log 1 h, and reopen Weekly Plan to show 6 remaining hours. Log another 7 h. The task now has 8 actual hours against a 7 h budget and remains open. **Revise remaining workload** can put additional planned time back on the schedule.
6. **Evaluation.** Open AI & ML Insights and compare the held-out MAE and RMSE with the baseline. Pick a scatter point, read its residual, and change one simulator input to inspect the learned relationship. State that the shipped metrics come from synthetic data.
7. **Feedback.** Return to the report and press **Confirm task is finished**. In Insights show its history row with a 4 h estimate, 7 h recommendation, and 8 h target. One completed task advances the private training count. Restart and sign in to verify persistence. Use the automated private-training test to demonstrate both candidate activation and retention without manufacturing records in a student's account.

## Testing and validation

```powershell
python -m unittest discover -s tests -v
```

**Verified on November 24, 2026. All 31 tests passed on Windows with Python 3.15.0.** A separate migration check compared the original account and task fields from the working database with a migrated copy. The field values matched, SQLite integrity passed, and the foreign-key check reported no errors.

The Windows suite exercises the original urgency boundaries, score and tie cases, greedy allocation, fractional hours, data isolation, concurrent writes, starter data, CRUD, authentication, and UI navigation. The two earlier database expectations that capped actual study time were updated to exercise the new over-budget behavior.

New tests verify the deterministic dataset, invalid rows and inputs, feature selection, fitting and persistence, negative-output clipping, independently recomputed holdout metrics, baseline comparison, missing/corrupt models, migration idempotence, completion history, acceptance/rejection/custom budgets, schedule updates, chronological personal training, candidate retention, and account isolation. Streamlit interaction tests exercise the user-facing workflow through its actual widgets. All test accounts use temporary databases.

## Limitations, improvements and conclusion

The linear model captures additive relationships learned from the training data. The demonstration generator favors this functional form. Real students may log time incompletely or finish tasks for reasons the three features cannot describe. With only 30 personal records, the minimum test split is six tasks and model comparisons can be noisy. A later evaluation could use prospectively collected, consented records and repeated time-ordered validation.

The scheduler uses one daily capacity value and relative calendar dates. Today's limit represents time available to plan from now. Overdue tasks need a revised deadline, and a greedy ordering can miss work that a different ordering could fit. Possible extensions include weekday-specific capacity, calibration by task type, uncertainty intervals and comparison with an exact scheduler on small instances.

StudyFlow lets a lecturer trace a student's estimate through a fitted model, a chosen budget, a priority score, and individual study allocations, then compare that plan with the independently recorded outcome.
