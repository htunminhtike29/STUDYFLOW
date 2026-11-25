"""Authenticated model evaluation, simulator, and controlled personal training."""
import pandas as pd
import streamlit as st

from ml_engine import FEATURES, MIN_PERSONAL

NAMES = dict(zip(FEATURES, ["Difficulty", "Days until deadline", "Original estimate"]))
TEAM = [("Htun Min Htike", "6709684"), ("Min Mg Mg Kyaw", "6709687"),
        ("Aung Khant Zaw", "6709532"), ("Pyae Sone Soe", "6709586"), ("Min Thant", "6805378")]


def about():
    with st.expander("About StudyFlow"):
        st.write("An Intelligent Study Scheduling System Using Machine Learning, Rule-Based Inference, and Heuristic Scheduling.")
        st.caption("University Artificial Intelligence group project")
        for name, student_id in TEAM:
            st.text(f"{name} · {student_id}")


def prediction_detail(model, result, estimate):
    meta = model.meta
    a, b, c = st.columns(3)
    a.metric("Original estimate", f"{estimate:g} h")
    b.metric("ML recommendation", f"{result['hours']:g} h")
    c.metric("Difference", f"{result['hours']-estimate:+.2f} h")
    st.caption(f"{meta['source']} · {meta['version']}. This is an estimate. Actual study time can vary.")
    if result["outside"]:
        st.warning("Outside the observed training range for " + ", ".join(NAMES[f] for f in result["outside"]) + ".")
    with st.expander("Explain this prediction"):
        contributions = [f"{NAMES[f]} {result['terms'][f]:+.3f} h" for f in FEATURES]
        st.write(f"Intercept {meta['intercept']:.3f} h · " + " · ".join(contributions))
        st.write(f"Raw regression output {result['raw']:.3f} h. Recommendations are bounded between "
                 f"0.1 and {meta['ceiling']:g} h and rounded to one decimal. The upper bound comes from training data.")
        st.caption("Each contribution is the feature value multiplied by its learned coefficient.")


def evaluation(meta):
    import plotly.express as px
    import plotly.graph_objects as go
    reg, baseline = meta["regression"], meta["baseline"]
    for col, key in zip(st.columns(4), ("MAE", "MSE", "RMSE", "R²")):
        value = reg[key]
        col.metric(key, f"{value:.3f}" if value is not None else "Undefined", border=True)
    st.caption("MAE is the average absolute error in hours. MSE averages squared errors in h². "
               "RMSE returns to hours and weights larger misses more heavily. R² measures explained "
               "variation and is undefined when all held-out targets are equal.")
    for m in ("MAE", "RMSE"):
        result = "lower" if reg[m] < baseline[m] else "higher" if reg[m] > baseline[m] else "equal"
        st.write(f"{m} is {result} compared with the baseline · Regression {reg[m]:.3f} h / original estimate {baseline[m]:.3f} h.")
    st.caption("Evaluation uses the same clipping and rounding as the interactive recommendation.")
    held = pd.DataFrame(meta["holdout"]).rename(columns={"actual": "Actual hours", "prediction": "Predicted hours"})
    held["Residual (h)"] = held["Actual hours"] - held["Predicted hours"]
    scatter = px.scatter(held, x="Actual hours", y="Predicted hours", title="Actual vs. predicted hours")
    low, high = held[["Actual hours", "Predicted hours"]].min().min(), held[["Actual hours", "Predicted hours"]].max().max()
    scatter.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color="#9693c9", dash="dot"))
    residuals = px.scatter(held, x="Predicted hours", y="Residual (h)", title="Residuals on held-out tasks")
    residuals.add_hline(y=0, line_dash="dot")
    coefs = px.bar(x=list(NAMES.values()), y=[meta["coefficients"][f] for f in FEATURES],
                   labels={"x": "Input", "y": "Hours per input unit"}, title="Learned coefficients")
    compare = go.Figure()
    for label, values in [("Linear regression", reg), ("Original estimate", baseline)]:
        compare.add_bar(name=label, x=["MAE", "RMSE"], y=[values["MAE"], values["RMSE"]])
    compare.update_layout(title="Model vs. baseline", yaxis_title="Error (h)", barmode="group")
    distribution = go.Figure()
    for name, values in [("Training", meta["training_hours"]), ("Testing", held["Actual hours"])]:
        distribution.add_histogram(name=name, x=values, opacity=.65, bingroup="hours", histnorm="percent")
    distribution.update_layout(title="Actual study hours by split", xaxis_title="Actual hours",
                               yaxis_title="Share of split (%)", barmode="overlay")
    a, b = st.columns(2)
    for container, chart in [(a, scatter), (b, residuals), (a, coefs), (b, compare), (st, distribution)]:
        chart.update_layout(height=300, margin=dict(l=12, r=12, t=48, b=16),
                            colorway=["#59d9df", "#b3a0ff"])
        container.plotly_chart(chart, width="stretch")


def insights(db, sid, model):
    st.title("AI & ML Insights")
    history = db.history(sid)
    count = sum(r["eligible"] for r in history)
    if model:
        meta = model.meta
        st.subheader("Workload prediction model")
        st.write(f"Multiple Linear Regression · Supervised learning · Predicts actual study hours")
        st.caption(f"{meta['source']} · {meta['version']} · Trained {meta['trained_at']} · {meta['split']}")
        a, b, c = st.columns(3)
        a.metric("Training tasks", meta["train_size"], border=True)
        b.metric("Held-out tasks", meta["test_size"], border=True)
        c.metric("Eligible personal records", count, border=True)
        if meta["source"] == "Synthetic demonstration":
            st.info("These results measure performance on generated academic tasks. Real-world accuracy requires a separate evaluation.")
        st.subheader("Held-out performance")
        evaluation(meta)
        st.subheader("Prediction simulator")
        a, b, c = st.columns(3)
        d = a.slider("Difficulty", 1, 5, 3)
        days = b.number_input("Days until deadline", value=7, step=1)
        estimate = c.number_input("Original estimate (hours)", min_value=.01, max_value=1_000_000., value=4.)
        prediction_detail(model, model.predict(d, days, estimate), estimate)
        st.code("Hours = " + f"{meta['intercept']:.4f} " + " ".join(
            f"{meta['coefficients'][f]:+.4f} × {NAMES[f]}" for f in FEATURES), language=None)
        st.caption("Each coefficient is the learned difference in predicted hours per unit increase, "
                   "holding the other inputs fixed. These associations do not establish causation.")
    st.subheader("Personal training history")
    st.write(f"{count} of {MIN_PERSONAL} required completed tasks. Only eligible records from this account "
             "are used. Training reserves the most recent 20% for evaluation. Logging time recalculates "
             "the plan immediately. Training starts only when you press the button.")
    if history:
        cols = ["subject_name", "difficulty", "days_until_deadline", "estimated_hours", "predicted_hours",
                "actual_hours", "completed_at", "eligible"]
        st.dataframe(pd.DataFrame(history)[cols], hide_index=True)
    st.caption("Sample tasks and unverified legacy creation inputs are excluded from personal training.")
    if st.button("Train Personal Model", disabled=count < MIN_PERSONAL, type="primary"):
        try:
            from train_model import train_personal
            _, accepted = train_personal(db, sid)
        except (ImportError, OSError, ValueError) as exc:
            st.error(f"Personal training could not finish. {exc}")
        else:
            st.session_state["notice"] = ("Personal model activated." if accepted else
                                          "Candidate saved for review. It did not improve both baseline errors.")
            st.rerun()
    reports = db.model_reports(sid)
    if reports:
        with st.expander("Personal model versions", expanded=True):
            st.dataframe([{"Version": m["version"], "Active": active, "Trained": m["trained_at"],
                           "MAE (h)": m["regression"]["MAE"], "Baseline MAE (h)": m["baseline"]["MAE"],
                           "RMSE (h)": m["regression"]["RMSE"], "Baseline RMSE (h)": m["baseline"]["RMSE"]}
                          for active, m in reports], hide_index=True)
