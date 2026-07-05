"""Greedy activity selection -- ST5003CEM Task 3 (greedy component).

Route-engine framing: a single delivery van (or one loading bay) can serve one
job at a time. Each job has a fixed [start, finish) window. Select the maximum
number of mutually non-overlapping jobs.

Greedy rule: sort by *earliest finish time* and repeatedly take the next job
whose start is >= the last taken job's finish.

Complexity: O(n log n) (the sort dominates; the scan is O(n)).

--------------------------------------------------------------------------
OPTIMALITY -- exchange argument (proved, not asserted):

Let g1, g2, ... be the activities the greedy picks (in order of selection), and
let o1, o2, ... be any optimal solution, both sorted by finish time.

Claim: f(g1) <= f(o1). Greedy picks the activity with the globally earliest
finish time, so its first choice finishes no later than the first choice of any
feasible solution, including the optimal one.

Exchange: replace o1 with g1 in the optimal solution. Because f(g1) <= f(o1),
g1 still finishes before o2 starts (o2 started after o1 finished >= g1 finished),
so the swapped set is still feasible and has the same size -> still optimal.

Induction: having fixed the first k greedy choices inside an optimal solution,
the same argument applied to the remaining sub-problem (activities starting
after f(gk)) shows greedy's (k+1)-th choice can replace the optimal's, keeping
feasibility and size. Hence greedy matches an optimal solution element by
element and is itself optimal.  QED.

This relies on the two hallmarks of greedy-amenable problems:
  * greedy-choice property  -- a globally optimal solution contains the locally
    greedy first choice (earliest finish);
  * optimal substructure     -- after committing a choice, an optimal solution to
    the remaining sub-problem completes an optimal whole.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

Activity = Tuple[int, int]  # (start, finish)


def select_activities(activities: List[Activity]) -> List[int]:
    """Return indices (into the original list) of a maximum compatible subset."""
    for s, f in activities:
        if f < s:
            raise ValueError(f"activity finish {f} precedes start {s}")
    order = sorted(range(len(activities)), key=lambda i: activities[i][1])
    chosen: List[int] = []
    last_finish = float("-inf")
    for i in order:
        s, f = activities[i]
        if s >= last_finish:
            chosen.append(i)
            last_finish = f
    return chosen


def select_activities_bruteforce(activities: List[Activity]) -> int:
    """O(2^n) max-compatible-subset size; validates greedy optimality on small n."""
    n = len(activities)
    best = 0
    for r in range(n, 0, -1):
        if r <= best:
            break
        for combo in combinations(range(n), r):
            iv = sorted((activities[i] for i in combo))
            if all(iv[k][1] <= iv[k + 1][0] for k in range(len(iv) - 1)):
                best = max(best, r)
                break
    return best
