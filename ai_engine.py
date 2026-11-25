"""Deterministic inference and exact rational scores for reproducible ordering."""
from dataclasses import dataclass
from fractions import Fraction

from models import Task, fmt

LABELS = ("", "Very low", "Low", "Moderate", "High", "Critical")


def infer_urgency(days):
    if days <= 1:
        return 5
    if days <= 3:
        return 4
    if days <= 5:
        return 3
    if days <= 7:
        return 2
    return 1


def normalize(remaining, maximum):
    return 5 * Fraction(remaining) / Fraction(maximum) if maximum > 0 else Fraction(0)


@dataclass(frozen=True)
class Priority:
    task: Task
    days: int
    urgency: int
    workload: Fraction
    score: Fraction

    @property
    def key(self):
        return (-self.score, self.task.deadline, -self.task.remaining, self.task.task_id)

    @property
    def status(self):
        return "Overdue" if self.days < 0 else LABELS[self.urgency]

    def explain(self):
        if self.days < 0:
            due = f"Overdue by {-self.days} day(s)"
        elif self.days == 0:
            due = "Due today"
        elif self.days == 1:
            due = "Due tomorrow"
        else:
            due = f"Due in {self.days} days"
        return (
            f"{due}, giving {LABELS[self.urgency].lower()} urgency ({self.urgency}/5) "
            f"and {0.5 * self.urgency:.2f} urgency points. Difficulty {self.task.difficulty}/5 "
            f"adds {0.3 * self.task.difficulty:.2f} points. The {fmt(self.task.remaining)} h left "
            f"normalize to {float(self.workload):.2f}/5 against your largest unfinished task, "
            f"adding {float(self.workload / 5):.2f} points. Total priority is {float(self.score):.2f}/5."
        )


def rank_tasks(tasks, today):
    active = [t for t in tasks if t.remaining > 0]
    maximum = max((t.remaining for t in active), default=0)
    ranked = []
    for task in active:
        days = (task.deadline - today).days
        urgency = infer_urgency(days)
        workload = normalize(task.remaining, maximum)
        score = Fraction(urgency, 2) + Fraction(3 * task.difficulty, 10) + workload / 5
        ranked.append(Priority(task, days, urgency, workload, score))
    return sorted(ranked, key=lambda item: item.key)
