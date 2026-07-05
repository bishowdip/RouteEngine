"""Parallel multi-source shortest paths -- ST5003CEM Task 5.

Workload: run our Task-2 Dijkstra from many independent source nodes. The
sources are independent, so this is embarrassingly parallel -- ideal for clean
speedup curves while reusing existing, validated code.

Three implementations of the SAME computation:
  * ``sequential_multi_source``     -- baseline.
  * ``threaded_multi_source``       -- a shared work queue + worker threads, with
    a ``threading.Lock`` guarding the shared results dict. Demonstrates correct
    synchronisation AND that CPU-bound Python threads do not speed up, because
    the Global Interpreter Lock serialises bytecode execution.
  * ``multiprocessing_multi_source``-- a process pool. Separate interpreters,
    separate GILs -> real parallelism and measurable speedup, at the cost of
    process-spawn + pickling/IPC overhead.

Correctness: the threaded/process results must equal the sequential results
exactly (no races / lost updates) -- asserted in the Task 5 tests and benchmark.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional

from src.graph.dijkstra import dijkstra
from src.graph.graph import Graph

# Reducers turn a source's full distance map into the value we keep. Returning a
# scalar (farness) instead of the whole dict is the key to making the
# multiprocessing version actually faster: it slashes the inter-process data
# that must be pickled back, so compute -- not IPC -- dominates.
def _full(dist: Dict[int, float]) -> Dict[int, float]:
    return dist


def _farness(dist: Dict[int, float]) -> float:
    """Sum of finite shortest-path distances from the source (closeness/farness
    centrality) -- a single float, and a genuinely useful 'best depot' metric."""
    inf = float("inf")
    return sum(d for d in dist.values() if d != inf)


REDUCERS: Dict[str, Callable[[Dict[int, float]], object]] = {
    "full": _full,
    "farness": _farness,
}


# --------------------------------------------------------------- sequential
def sequential_multi_source(graph: Graph, sources: List[int], reduce: str = "full") -> Dict[int, object]:
    """Baseline: one Dijkstra per source, in order."""
    red = REDUCERS[reduce]
    return {s: red(dijkstra(graph, s)[0]) for s in sources}


# ----------------------------------------------------------------- threading
def threaded_multi_source(
    graph: Graph, sources: List[int], n_workers: int = 4, reduce: str = "full"
) -> Dict[int, object]:
    """Worker threads pull sources off a shared queue; a Lock guards results.

    The Lock makes the shared-dict update a critical section -- the textbook
    race-condition guard. (CPython dict writes are individually atomic, but we
    use the Lock to demonstrate correct synchronisation explicitly, as the brief
    requires.)
    """
    red = REDUCERS[reduce]
    work: "queue.Queue[int]" = queue.Queue()
    for s in sources:
        work.put(s)

    results: Dict[int, object] = {}
    results_lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                s = work.get_nowait()
            except queue.Empty:
                return
            value = red(dijkstra(graph, s)[0])     # the CPU-bound region
            with results_lock:                     # critical section
                results[s] = value
            work.task_done()

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ------------------------------------------------------------ multiprocessing
# A pool initializer stashes the graph in a per-process global so we pickle the
# (large) graph once per worker rather than once per source.
_WORKER_GRAPH: Graph | None = None
_WORKER_REDUCE: Callable[[Dict[int, float]], object] = _full


def _init_worker(graph: Graph, reduce: str) -> None:
    global _WORKER_GRAPH, _WORKER_REDUCE
    _WORKER_GRAPH = graph
    _WORKER_REDUCE = REDUCERS[reduce]


def _solve_one(source: int) -> tuple[int, object]:
    assert _WORKER_GRAPH is not None
    return source, _WORKER_REDUCE(dijkstra(_WORKER_GRAPH, source)[0])


def multiprocessing_multi_source(
    graph: Graph, sources: List[int], n_workers: int = 4, reduce: str = "full",
    chunksize: int = 4, start_method: Optional[str] = None,
) -> Dict[int, object]:
    """Process pool: separate interpreters escape the GIL for true parallelism.

    ``chunksize`` batches several sources per dispatch to amortise task overhead.
    ``start_method`` selects the process-creation strategy:
      * "fork" (POSIX): children inherit the parent's memory copy-on-write, so the
        graph need not be re-imported or pickled into each worker -- low startup
        cost, the regime where speedup is clean.
      * "spawn" (macOS/Windows default): a fresh interpreter re-imports modules
        and the graph is pickled across the pipe -- safe but high startup cost.
    The graph is pickled once per worker (via the initializer), not per source.
    """
    ctx = multiprocessing.get_context(start_method) if start_method else None
    results: Dict[int, object] = {}
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
        initializer=_init_worker, initargs=(graph, reduce)
    ) as pool:
        for source, value in pool.map(_solve_one, sources, chunksize=chunksize):
            results[source] = value
    return results


def results_equal(a: Dict[int, object], b: Dict[int, object]) -> bool:
    """Exact equality of two result sets -- the correctness / no-race check."""
    if a.keys() != b.keys():
        return False
    for s in a:
        if a[s] != b[s]:
            return False
    return True
