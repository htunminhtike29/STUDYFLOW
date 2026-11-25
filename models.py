"""Shared task record and hour validation at the SQLite boundary."""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation


def hours(value, *, daily=False):
    try:
        amount = Decimal(str(value))
    except (ValueError, InvalidOperation):
        raise ValueError("Enter a number of study hours.") from None
    maximum = 24 if daily else 1_000_000
    if not amount.is_finite() or not 0 < amount <= maximum:
        raise ValueError(f"Hours must be greater than zero and at most {maximum:,}.")
    if amount != amount.quantize(Decimal("0.000001")):
        raise ValueError("Use up to six decimal places for study hours.")
    return amount


def fmt(value):
    value = Decimal(str(value))
    return format(value.normalize(), "f")


@dataclass(frozen=True)
class Task:
    task_id: int
    subject_name: str
    deadline: date
    difficulty: int
    total_hours: Decimal
    completed_hours: Decimal = Decimal(0)

    estimated_hours: Decimal | None = None
    predicted_hours: Decimal | None = None
    finished: bool = False
    budget_source: str = "Original estimate"
    model_version: str | None = None

    @property
    def remaining(self):
        return Decimal(0) if self.finished else max(Decimal(0), self.total_hours - self.completed_hours)
