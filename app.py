"""StudyFlow's Streamlit entry point. Run from this directory with streamlit run app.py."""
import os
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from html import escape
from pathlib import Path

import streamlit as st

import auth
from insights import about, insights, prediction_detail
from ml_engine import available_model
from ai_engine import LABELS, rank_tasks
from database import DEFAULT_PATH, Database, validate_task
from models import fmt
from scheduler import schedule

ROOT = Path(__file__).resolve().parent
SECTIONS = ["Dashboard", "Subjects & Tasks", "Weekly Plan", "Progress Tracker", "AI & ML Insights"]
BRAND = """<div class="sf-brand"><span class="sf-mark" aria-hidden="true">S</span>
<div><strong>STUDYFLOW</strong><small>Hybrid AI Smart Study Planner</small></div></div>"""
st.set_page_config(page_title="StudyFlow · Hybrid AI Smart Study Planner", page_icon=":material/menu_book:",
                   layout="wide")
st.html(ROOT / "assets" / "style.css")


def saved(message):
    st.session_state["notice"] = message
    st.rerun()


def notice():
    if message := st.session_state.pop("notice", None):
        st.success(message)


def duration(label, value=1.0, *, daily=False, key=None):
    return st.number_input(label, min_value=0.000001, max_value=24.0 if daily else 1_000_000.0,
                           value=float(value), step=0.5, format="%.6f", key=key)


def login_page(db):
    cover, form = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with cover:
        st.html("""<section class="sf-cover"><span class="sf-wordmark">StudyFlow</span>
        <div><h1>Hybrid AI Smart<br><em>Study Planner</em></h1>
        <p>Plan your study week around deadlines, difficulty, and the hours you have.</p></div>
        <div class="sf-weights" aria-label="Priority weights"><span><b>50%</b>Urgency</span>
        <span><b>30%</b>Difficulty</span><span><b>20%</b>Workload</span></div></section>""")
    with form:
        notice()
        mode = st.radio("Account access", ["Sign in", "Register"], horizontal=True,
                        label_visibility="collapsed", key="access")
        creating = mode == "Register"
        st.subheader("Create your account" if creating else "Welcome back")
        st.caption("Start with three sample subjects." if creating else "Sign in to your study planner.")
        with st.form("register" if creating else "login"):
            sid = st.text_input("Student ID", max_chars=64)
            if creating:
                name = st.text_input("Full name", max_chars=100)
            password = st.text_input("Password", type="password", max_chars=256,
                                     help="Use 8–256 characters." if creating else None)
            if creating:
                confirmation = st.text_input("Confirm password", type="password", max_chars=256)
                limit = duration("Daily study limit (hours)", 3, daily=True)
            submitted = st.form_submit_button("Create account" if creating else "Sign in",
                                              type="primary", width="stretch")
        if submitted:
            if creating:
                auth.register(db, sid, name, password, confirmation, limit)
                st.session_state.clear()
                saved("Account created. Sign in to see your sample subjects.")
            elif auth.authenticate(db, sid, password):
                st.session_state.clear()
                st.session_state["student_id"] = sid.strip()
                st.rerun()
            else:
                st.error("Check your student ID and password, then try again.")


def formula():
    with st.expander("How AI Prioritization Works"):
        st.markdown("**Priority = 0.50 × urgency + 0.30 × difficulty + 0.20 × normalized workload**")
        st.write("Workload = 5 × remaining task hours / largest unfinished workload.")
        st.table([{"Days until due": span, "Urgency": f"{i}/5 · {LABELS[i]}"}
                  for span, i in [("Overdue, today, or tomorrow", 5), ("2–3", 4), ("4–5", 3),
                                  ("6–7", 2), ("8 or more", 1)]])
        st.write("The queue compares exact scores, then earlier deadlines, more remaining hours, "
                 "and finally lower task IDs. Overdue work needs a new deadline. The greedy plan "
                 "prefers up to two hours per subject each day while other eligible work can use the time.")


def reason(item, ranks):
    st.write(item.explain())
    if item.days < 0:
        st.caption("This deadline has passed. Edit the deadline to return this task to the schedule.")
    else:
        eligible = [r for r in ranks if r.days >= 0]
        pos = eligible.index(item)
        detail = f"Queue position {pos + 1} of {len(eligible)} eligible tasks."
        if pos + 1 < len(eligible):
            other = eligible[pos + 1]
            labels = ["a higher unrounded score", "an earlier deadline", "more remaining hours", "a lower task ID"]
            difference = next(label for a, b, label in zip(item.key, other.key, labels) if a != b)
            detail += f" It precedes {other.task.subject_name} because it has {difference}."
        st.caption(detail + " Daily capacity, deadlines, and the subject preference determine the actual blocks.")


def sidebar(db, student):
    with st.sidebar:
        st.html(BRAND)
        st.text(student["name"])
        st.caption(f"Student ID · {student['student_id']}")
        st.divider()
        page = st.radio("Workspace", SECTIONS, key="view")
        st.divider()
        with st.expander(f"Daily study limit · {fmt(student['daily_limit'])} h"):
            with st.form("capacity"):
                limit = duration("Hours per day", student["daily_limit"], daily=True)
                if st.form_submit_button("Save limit"):
                    db.set_limit(student["student_id"], limit)
                    saved("Daily limit saved. Your study plan has been recalculated.")
        about()
        if st.button("Logout", icon=":material/logout:"):
            auth.logout(st.session_state)
            st.rerun()
    return page


def overall(tasks):
    total = sum(t.total_hours for t in tasks)
    credited = sum(t.total_hours - t.remaining for t in tasks)
    fraction = float(credited / total) if total else 0.0
    finished = sum(t.finished for t in tasks)
    st.progress(fraction, text=f"Planned workload covered · {fraction:.1%} · {finished} of {len(tasks)} tasks complete")


def metrics(values):
    for col, (label, value) in zip(st.columns(len(values)), values):
        col.metric(label, value, border=True)


def dashboard(tasks, ranked, student, today, model):
    hour = datetime.now().hour
    greeting = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    st.html(f"""<header class="sf-banner"><span class="sf-date">{today:%A, %d %B %Y}</span>
            <h1>Good {greeting}, {escape(student['name'])}!</h1></header>""")
    metrics([("Active tasks", sum(not t.finished for t in tasks)), ("Hours remaining", fmt(sum(t.remaining for t in tasks))),
             ("Daily limit", f"{fmt(student['daily_limit'])} h"), ("Completed", sum(t.finished for t in tasks)),
             ("Due in next 7 days", sum(not t.finished and 0 <= (t.deadline-today).days < 7 for t in tasks))])
    overall(tasks)
    st.subheader("Hybrid AI")
    st.caption("Linear regression recommends a budget. Deadline rules and the priority queue turn your chosen hours into a seven-day plan.")
    st.write(f"Active model · {model.meta['source'] if model else 'Unavailable'}")
    plan = schedule(ranked, today, student["daily_limit"])
    compared = [t for t in tasks if t.predicted_hours is not None]
    metrics([("ML budgets in use", sum(t.budget_source == "ML recommendation" for t in tasks)),
             ("Original estimates¹", f"{fmt(sum(t.estimated_hours for t in compared))} h"),
             ("ML recommendations¹", f"{fmt(sum(t.predicted_hours for t in compared))} h"),
             ("Unmet this week", f"{fmt(sum(plan.unmet.values()))} h")])
    st.caption("¹ Compared across tasks with a saved prediction.")
    exhausted = [t for t in tasks if not t.finished and not t.remaining]
    if exhausted:
        st.warning("Open tasks with exhausted budgets: " + ", ".join(t.subject_name for t in exhausted)
                   + ". Revise the budget or confirm completion in Progress Tracker.")
    if not ranked:
        st.info("No planned hours remain." if tasks else "Add your first task in Subjects & Tasks.")
        formula()
        return
    st.subheader("Top priorities")
    for col, item in zip(st.columns(min(3, len(ranked))), ranked[:3]):
        col.html(f"""<article class="sf-focus"><span class="sf-date">{item.status} · {float(item.score):.2f}/5</span>
                 <strong>{escape(item.task.subject_name)}</strong>
                 <small>Due {item.task.deadline:%d %b} · {fmt(item.task.remaining)} h remaining</small></article>""")
    st.subheader("Upcoming deadlines")
    rows = [{"Subject": r.task.subject_name, "Deadline": r.task.deadline, "Days remaining": r.days,
             "Difficulty": r.task.difficulty, "Remaining hours": float(r.task.remaining),
             "Priority": f"{float(r.score):.2f}", "Urgency": r.status} for r in
            sorted(ranked, key=lambda r: (r.task.deadline, r.key))]
    st.dataframe(rows, hide_index=True, alt="Unfinished tasks ordered by deadline")
    st.subheader("Workload by deadline")
    workload = defaultdict(Decimal)
    for r in ranked:
        workload[r.task.deadline] += r.task.remaining
    st.bar_chart([{"Due date": d.isoformat(), "Hours": float(v)} for d, v in sorted(workload.items())],
                 x="Due date", y="Hours", color="#625ab9", height=190, alt="Remaining hours by due date")
    st.subheader("Priority explanations")
    for item in ranked:
        with st.expander(f"{item.task.subject_name} · #{item.task.task_id}"):
            reason(item, ranked)
    formula()


def task_fields(task=None):
    name = st.text_input("Subject or task name", value=task.subject_name if task else "", max_chars=160)
    a, b, c = st.columns(3)
    deadline = a.date_input("Deadline", value=task.deadline if task else date.today(),
                            min_value=date(1900, 1, 1), max_value=date(9999, 12, 31))
    difficulty = b.selectbox("Difficulty (1–5)", range(1, 6), index=task.difficulty-1 if task else 2)
    with c:
        estimate = duration("Accepted budget (hours)" if task else "Estimated study hours", task.total_hours if task else 1.0)
    return name, deadline, difficulty, estimate


def choose(tasks, key):
    lookup = {t.task_id: t for t in tasks}
    choice = st.selectbox("Select a task", list(lookup), key=key,
                          format_func=lambda tid: f"{lookup[tid].subject_name} · #{tid}")
    return lookup[choice]


def task_page(db, sid, tasks, ranked, today, model):
    st.title("Subjects & Tasks")
    with st.expander("Add a task", expanded=not tasks):
        fields = task_fields()
        name, deadline, difficulty, estimate = fields
        signature = (fields, today, model.meta["version"] if model else None)
        if st.button("Predict study hours", disabled=model is None):
            validate_task(*fields)
            st.session_state["preview"] = signature
        preview = st.session_state.get("preview") == signature and model is not None
        options = ["Original estimate", "Custom budget"]
        prediction = None
        if preview:
            prediction = model.predict(difficulty, (deadline-today).days, estimate)
            prediction_detail(model, prediction, estimate)
            options.insert(1, "ML recommendation")
        source = st.radio("Planning budget", options, horizontal=True)
        budget = duration("Custom budget (hours)", estimate) if source == "Custom budget" else None
        if st.button("Add task", type="primary"):
            db.add_task(sid, *fields, budget=budget, predicted=prediction["hours"] if preview else None,
                        version=model.meta["version"] if preview else None, source=source, today=today)
            st.session_state.pop("preview", None)
            saved("Task added.")
    priorities = {r.task.task_id: r for r in ranked}
    if tasks:
        rows = []
        for t in tasks:
            r = priorities.get(t.task_id)
            rows.append({"ID": t.task_id, "Subject": t.subject_name, "Deadline": t.deadline,
                         "Difficulty": t.difficulty, "Accepted hours": float(t.total_hours),
                         "Original estimate": float(t.estimated_hours), "Budget source": t.budget_source,
                         "Actual hours": float(t.completed_hours), "Remaining hours": float(t.remaining),
                         "Priority": f"{float(r.score):.2f}" if r else "—",
                         "Status": r.status if r else ("Completed" if t.finished else "Budget exhausted")})
        st.dataframe(rows, hide_index=True, alt="Tasks and recorded progress")
        t = choose(tasks, "manage_id")
        if t.task_id in priorities:
            with st.expander("View AI priority explanation"):
                reason(priorities[t.task_id], ranked)
        with st.expander("Edit selected task"):
            with st.form(f"edit_{t.task_id}"):
                fields = task_fields(t)
                if st.form_submit_button("Save changes", type="primary"):
                    db.edit_task(sid, t.task_id, *fields)
                    saved("Task updated.")
        with st.expander("Delete selected task"):
            confirm = st.checkbox(f"Delete {t.subject_name} and its progress", key=f"confirm_{t.task_id}")
            if st.button("Delete task", disabled=not confirm):
                db.delete_task(sid, t.task_id)
                saved("Task deleted.")
    else:
        st.info("Your task list is empty. Add a task above to begin.")
    with st.expander("Manage sample tasks"):
        st.write("Remove the three starter tasks or restore them with fresh deadlines and zero progress. "
                 "These actions apply only to tasks originally provided as samples.")
        confirmed = st.checkbox("Confirm changes to sample tasks")
        a, b = st.columns(2)
        if a.button("Remove sample tasks", disabled=not confirmed):
            db.samples(sid, restore=False, today=today)
            saved("Sample tasks removed.")
        if b.button("Restore original samples", disabled=not confirmed):
            db.samples(sid, restore=True, today=today)
            saved("Original sample dataset restored.")


def weekly(ranked, plan, limit, tasks):
    st.title("Weekly Plan")
    st.caption(f"{plan.days[0].date:%d %b} – {plan.days[-1].date:%d %b %Y} · Seven-day greedy study schedule")
    lookup = {r.task.task_id: r for r in ranked}
    metrics([("Scheduled", f"{fmt(plan.total)} h"), ("Unmet this week", f"{fmt(sum(plan.unmet.values()))} h"),
             ("Beyond this week", f"{fmt(sum(plan.later.values()))} h"),
             ("Overdue", f"{fmt(sum(plan.overdue.values()))} h")])
    for tid, amount in plan.unmet.items():
        t = lookup[tid].task
        st.warning(f"{t.subject_name} · Due {t.deadline:%d %b}. The plan leaves {fmt(amount)} h unmet "
                   "because available slots through the deadline are filled. Review your limit or deadline.")
    for tid, amount in plan.overdue.items():
        t = lookup[tid].task
        st.warning(f"{t.subject_name} · Overdue since {t.deadline:%d %b}. {fmt(amount)} h remain. "
                   "Choose a new deadline in Subjects & Tasks to schedule this work.")
    if plan.later:
        with st.expander("Work remaining beyond this week"):
            for tid, amount in plan.later.items():
                t = lookup[tid].task
                st.text(f"{t.subject_name} · {fmt(amount)} h remaining · Due {t.deadline:%d %b}")
    exhausted = [t.subject_name for t in tasks if not t.finished and t.remaining == 0]
    if exhausted:
        st.warning("No planned time remains for these open tasks: " + ", ".join(exhausted)
                   + ". Update their budgets in Progress Tracker to schedule more time.")
    by_subject = defaultdict(Decimal)
    st.subheader("Daily schedule")
    for day in plan.days:
        with st.expander(f"{day.date:%A, %d %b}  ·  {fmt(day.total)} / {fmt(limit)} h",
                         expanded=day is plan.days[0]):
            if not day.blocks:
                st.caption("No study blocks scheduled.")
            for tid, amount in day.blocks.items():
                t = lookup[tid].task
                by_subject[t.subject_name] += amount
                st.text(f"{t.subject_name} · #{tid}   ·   {fmt(amount)} h")
                st.caption(f"{t.budget_source} · Original {fmt(t.estimated_hours)} h · "
                           f"Budget {fmt(t.total_hours)} h · Remaining {fmt(t.remaining)} h")
            st.caption(f"{fmt(limit-day.total)} h available · {day.total / limit:.1%} of daily capacity used")
    a, b = st.columns(2)
    with a:
        st.subheader("Scheduled hours by day")
        st.bar_chart([{"Day": d.date.isoformat(), "Hours": float(d.total)} for d in plan.days],
                     x="Day", y="Hours", height=220, color="#625ab9", alt="Planned hours for each of seven days")
    with b:
        st.subheader("Scheduled hours by subject")
        if by_subject:
            st.bar_chart([{"Subject": k, "Hours": float(v)} for k, v in by_subject.items()],
                         x="Subject", y="Hours", horizontal=True, height=220, color="#467f83",
                         alt="Planned hours grouped by subject")
        else:
            st.caption("Scheduled subjects will appear here.")
    st.subheader("Daily capacity utilization")
    st.dataframe([{"Day": d.date, "Scheduled (h)": float(d.total), "Available (h)": float(limit-d.total),
                   "Utilization (%)": float(100*d.total/limit)} for d in plan.days],
                 column_config={"Utilization (%)": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                                   format="%.1f%%")},
                 hide_index=True, alt="Daily capacity and utilization")
    scheduled_ids = {tid for d in plan.days for tid in d.blocks}
    if scheduled_ids:
        st.subheader("Why these tasks?")
        for r in ranked:
            if r.task.task_id in scheduled_ids:
                with st.expander(f"{r.task.subject_name} · #{r.task.task_id}"):
                    reason(r, ranked)
    formula()


def progress(db, sid, tasks):
    st.title("Progress Tracker")
    overall(tasks)
    active = [t for t in tasks if not t.finished]
    if active:
        with st.container(border=True):
            st.subheader("Log a study session")
            t = choose(active, "log_id")
            st.caption(f"{fmt(t.remaining)} hours remaining for this task.")
            a, b = st.columns(2)
            for col, amount in [(a, 0.5), (b, 1.0)]:
                if col.button(f"+{amount:.1f} hour", width="stretch"):
                    db.log_progress(sid, t.task_id, amount)
                    saved(f"Logged {amount:.1f} h. Your priorities and plan have been updated.")
            with st.form(f"custom_{t.task_id}"):
                amount = duration("Custom study duration (hours)", 0.5)
                if st.form_submit_button("Log custom duration", type="primary"):
                    db.log_progress(sid, t.task_id, amount)
                    saved(f"Logged {fmt(amount)} h. Your priorities and plan have been updated.")
            if not t.remaining:
                st.warning(f"Budget exhausted · {fmt(t.completed_hours)} h logged against {fmt(t.total_hours)} h planned. "
                           "You can keep logging, add more planned time, or confirm completion.")
                with st.form(f"revise_{t.task_id}"):
                    extra = duration("Additional hours still needed", 1.)
                    if st.form_submit_button("Revise remaining workload"):
                        db.revise_remaining(sid, t.task_id, extra)
                        saved("Remaining workload revised.")
            if st.button("Confirm task is finished", disabled=t.completed_hours <= 0):
                db.finish(sid, t.task_id)
                saved("Task marked complete.")
    else:
        st.info("All tasks are complete." if tasks else "Add a task to begin tracking study time.")
    if tasks:
        st.subheader("Progress by task")
        for t in tasks:
            with st.container(border=True):
                st.text(f"{t.subject_name} · #{t.task_id}")
                fraction = float(t.completed_hours / t.total_hours)
                st.progress(min(1., fraction), text=f"{fraction:.1%} of budget logged · {fmt(t.completed_hours)} actual hours")
                predicted = fmt(t.predicted_hours) + " h" if t.predicted_hours is not None else "None"
                st.caption(f"Original {fmt(t.estimated_hours)} h · ML prediction {predicted} · Budget {fmt(t.total_hours)} h · "
                           f"Remaining {fmt(t.remaining)} h · {'Completed' if t.finished else 'Open'}")
    sessions = db.sessions(sid)
    if sessions:
        with st.expander("Recorded study sessions"):
            st.dataframe(sessions, hide_index=True)


def main():
    db = Database(os.environ.get("STUDYFLOW_DB", DEFAULT_PATH))
    db.initialize()
    sid = st.session_state.get("student_id")
    if sid is None:
        login_page(db)
        return
    student = db.student(sid)
    if student is None:
        auth.logout(st.session_state)
        saved("Please sign in again.")
    today = date.today()
    page = sidebar(db, student)
    notice()
    tasks = db.tasks(sid)
    ranked = rank_tasks(tasks, today)
    limit = Decimal(str(student["daily_limit"]))
    model, message = available_model(db, sid)
    if message:
        st.warning(message)
    if page == "Dashboard":
        dashboard(tasks, ranked, student, today, model)
    elif page == "Subjects & Tasks":
        task_page(db, sid, tasks, ranked, today, model)
    elif page == "Weekly Plan":
        weekly(ranked, schedule(ranked, today, limit), limit, tasks)
    elif page == "Progress Tracker":
        progress(db, sid, tasks)
    else:
        insights(db, sid, model)


try:
    main()
except ValueError as error:
    st.error(str(error))
except sqlite3.Error:
    st.error("StudyFlow could not complete the database operation. Check that the database folder "
             "is writable, then retry. Any unfinished transaction has been rolled back.")
