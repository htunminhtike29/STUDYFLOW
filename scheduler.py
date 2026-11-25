"""Seven-day greedy allocation. The caller supplies a freshly ranked snapshot."""
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from models import hours

BLOCK = Decimal("0.5")
SUBJECT_LIMIT = Decimal("2")


@dataclass
class Day:
    date: date
    blocks: dict[int, Decimal] = field(default_factory=dict)

    @property
    def total(self):
        return sum(self.blocks.values(), Decimal(0))


@dataclass
class Plan:
    days: list[Day]
    unmet: dict[int, Decimal]
    later: dict[int, Decimal]
    overdue: dict[int, Decimal]

    @property
    def total(self):
        return sum((day.total for day in self.days), Decimal(0))


def schedule(ranked, today, daily_limit):
    limit = hours(daily_limit, daily=True)
    days = [Day(today + timedelta(days=i)) for i in range(7)]
    by_id = {item.task.task_id: item for item in ranked}
    left = {tid: item.task.remaining for tid, item in by_id.items()}
    for day in days:
        free = limit
        by_subject = defaultdict(Decimal)
        queue = [item.key for tid, item in by_id.items()
                 if left[tid] > 0 and item.task.deadline >= day.date]
        heapq.heapify(queue)
        deferred = []
        relaxed = False
        while free > 0 and (queue or deferred):
            if not queue:
                # All remaining candidates have reached their subject preference.
                queue, deferred = deferred, []
                heapq.heapify(queue)
                relaxed = True
            entry = heapq.heappop(queue)
            tid = entry[-1]
            task = by_id[tid].task
            subject = task.subject_name.strip().casefold()
            future_days = (task.deadline - day.date).days
            required_today = max(Decimal(0), left[tid] - future_days * limit)
            # Allow urgent work through the soft cap if waiting would miss its deadline.
            allowance = free if relaxed else max(SUBJECT_LIMIT - by_subject[subject], required_today)
            if allowance <= 0:
                deferred.append(entry)
                continue
            amount = min(BLOCK, left[tid], free, allowance)
            day.blocks[tid] = day.blocks.get(tid, Decimal(0)) + amount
            by_subject[subject] += amount
            left[tid] -= amount
            free -= amount
            if left[tid] > 0:
                heapq.heappush(queue, entry)

    unmet, later, overdue = {}, {}, {}
    for tid, amount in left.items():
        if amount <= 0:
            continue
        deadline = by_id[tid].task.deadline
        if deadline < today:
            overdue[tid] = amount
        elif deadline <= days[-1].date:
            unmet[tid] = amount
        else:
            later[tid] = amount
    return Plan(days, unmet, later, overdue)
