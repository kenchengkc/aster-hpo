"""Owned spawn workers with isolated channels and bounded group shutdown.

Parent exchange threads handle blocking pickle/pipe I/O; the coordinator polls
only local mailboxes. A terminated worker's channel is never reused.
"""

import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext

from .core import _evaluate, _positive_integer


def _worker_loop(objective, connection, threads_per_worker=None):
    # The coordinator handles Ctrl-C and controls worker cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    bound_run = None
    try:
        while (request := connection.recv()) is not None:
            kind, run_id, *payload = request
            if kind == "release":
                if run_id == bound_run:
                    objective, bound_run = None, None
                continue
            attempt, binding, config, replication = payload
            try:
                if run_id != bound_run:
                    if binding is None:
                        raise RuntimeError("Worker received an unbound run")
                    objective, bound_run = pickle.loads(binding), run_id
                # Apply limits after unpickling too: the objective's imports may
                # have loaded native pools since this child first started.
                if threads_per_worker is not None:
                    from .runtime import thread_limits

                    control = thread_limits(threads_per_worker)
                else:
                    control = nullcontext()
                with control:
                    observation = _evaluate(objective, config, replication)
            except BaseException as exc:
                connection.send(("fatal", run_id, attempt, f"{type(exc).__name__}: {exc}"))
                return
            connection.send(("result", run_id, attempt, observation))
            del observation, binding, config, replication
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class Worker:
    def __init__(self, objective, wake, *, threads_per_worker=None):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_worker_loop,
            args=(objective, child, threads_per_worker),
            name="aster-worker",
        )
        self.job = None
        self.dispatched = None
        self.run_id = None
        self.bound_run = None
        self.requests = queue.SimpleQueue()
        self.responses = queue.SimpleQueue()
        self.wake = wake
        self.thread = threading.Thread(target=self._exchange, name="aster-worker-io", daemon=True)
        try:
            self.process.start()
            self.thread.start()
        except BaseException:
            if self.process.pid is not None:
                stop_workers([self], force=True)
            else:
                self.connection.close()
                self.process.close()
            raise
        finally:
            child.close()

    def _exchange(self):
        run_id = None
        try:
            while True:
                request = self.requests.get()
                if request is not None:
                    run_id = request[1]
                self.connection.send(request)
                if request is None:
                    return
                if request[0] == "release":
                    continue
                response = self.connection.recv()
                # Judge the deadline at receipt, even if other worker cleanup
                # delays the coordinator's inspection of this mailbox.
                self.responses.put((*response, time.perf_counter()))
                self.wake.set()
        except (EOFError, OSError) as exc:
            self.responses.put(
                ("lost", run_id, None, f"{type(exc).__name__}: {exc}", time.perf_counter())
            )
            self.wake.set()
        except BaseException as exc:
            self.responses.put(
                ("fatal", run_id, None, f"{type(exc).__name__}: {exc}", time.perf_counter())
            )
            self.wake.set()
        finally:
            self.connection.close()

    def submit(self, job, dispatched, *, run_id=None, objective=None):
        self.job, self.dispatched, self.run_id = job, dispatched, run_id
        trial, replication, attempt = job
        binding = None
        if self.bound_run != run_id:
            binding = pickle.dumps(objective)
            self.bound_run = run_id
        self.requests.put(("evaluate", run_id, attempt, binding, trial.config, replication))

    def release(self, run_id):
        if self.bound_run == run_id:
            self.requests.put(("release", run_id))
            self.bound_run = None
        self.job, self.dispatched, self.run_id = None, None, None

    def poll(self):
        try:
            return self.responses.get_nowait()
        except queue.Empty:
            return None


def stop_workers(workers, *, force):
    """Stop/reap the entire group with shared grace deadlines, never N × grace."""
    if not workers:
        return
    for worker in workers:
        worker.requests.put(None)
        if force and worker.process.is_alive():
            worker.process.terminate()
    deadline = time.perf_counter() + 0.2
    for worker in workers:
        worker.process.join(timeout=max(0, deadline - time.perf_counter()))
    for worker in workers:
        if worker.process.is_alive():
            worker.process.kill()
    deadline = time.perf_counter() + 1.0
    for worker in workers:
        worker.process.join(timeout=max(0, deadline - time.perf_counter()))
    alive = [worker.process.pid for worker in workers if worker.process.is_alive()]
    if alive:
        raise RuntimeError(f"Managed workers did not exit after kill: {alive}")
    deadline = time.perf_counter() + 0.2
    for worker in workers:
        if worker.thread.ident is not None:
            worker.thread.join(timeout=max(0, deadline - time.perf_counter()))
        worker.connection.close()
        worker.process.close()
    if any(worker.thread.is_alive() for worker in workers):
        raise RuntimeError("Worker I/O did not finish during cleanup")


class WorkerPool:
    """Explicitly owned spawn processes reusable across optimize and audit calls.

    A study may use up to ``workers`` processes with matching native thread limits.
    Calls sharing a pool must run sequentially. Callables are freshly serialized
    per invocation; module globals and library caches persist inside each child.
    Use a context manager or call close() to release idle processes and GPU memory.
    """

    def __init__(self, workers=1, threads_per_worker=1):
        _positive_integer(workers, "workers")
        _positive_integer(threads_per_worker, "threads_per_worker")
        self._capacity, self._threads = workers, threads_per_worker
        self._owner_pid = os.getpid()
        self._id = uuid.uuid4().hex
        self._children = []
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._borrowed = False
        self._closed = False
        self._run_pids = set()

    @property
    def workers(self):
        return self._capacity

    @property
    def threads_per_worker(self):
        return self._threads

    @property
    def closed(self):
        return self._closed

    @property
    def worker_pids(self):
        self._check_owner()
        return tuple(w.process.pid for w in self._children if w.process.is_alive())

    def _check_owner(self):
        if os.getpid() != self._owner_pid:
            raise RuntimeError("WorkerPool belongs to a different process")

    def __reduce__(self):
        raise TypeError("WorkerPool cannot be pickled")

    def __enter__(self):
        self._check_owner()
        if self.closed:
            raise RuntimeError("WorkerPool is closed")
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _lease(self, study):
        self._check_owner()
        with self._lock:
            if self.closed:
                raise RuntimeError("WorkerPool is closed")
            if self._borrowed:
                raise RuntimeError("WorkerPool is already in use")
            if study.workers > self.workers:
                raise ValueError("Study workers exceed WorkerPool capacity")
            if study.threads_per_worker != self.threads_per_worker:
                raise ValueError("Study threads_per_worker must match WorkerPool")
            self._borrowed = True
            self._run_pids = set()
        try:
            yield uuid.uuid4().hex
        except BaseException as exc:
            # Opening/recovering storage may fail before run_parallel takes
            # ownership. Such failures must reap previously idle children too.
            try:
                self._stop(force=True)
            except BaseException as cleanup:
                exc.add_note(f"WorkerPool cleanup also failed: {type(cleanup).__name__}: {cleanup}")
            raise
        finally:
            with self._lock:
                self._borrowed = False

    def close(self):
        """Reap all owned workers; closing during an active call is rejected."""
        self._check_owner()
        with self._lock:
            if self._borrowed:
                raise RuntimeError("WorkerPool is in use; cannot close a running pool")
            self._closed = True
            # Serialize the entire bounded cleanup, including process.close(),
            # so simultaneous close calls cannot operate on the same handles.
            self._stop(force=False)

    def _stop(self, *, force):
        stop_workers(self._children, force=force)
        self._children.clear()

    def _metadata(self, run_id):
        return {
            "worker_pool_id": self._id,
            "worker_pool_run_id": run_id,
            "worker_pool_capacity": self.workers,
            "worker_pool_pids": sorted(self._run_pids),
        }


def _retire_idle(workers):
    for worker in list(workers):
        if worker.job is not None:
            continue
        unhealthy = False
        while (response := worker.poll()) is not None:
            unhealthy |= response[0] in {"lost", "fatal"}
        if unhealthy or worker.process.exitcode is not None:
            workers.remove(worker)
            stop_workers([worker], force=True)


def run_parallel(coordinator, objective, limits, *, worker_pool=None, run_id=None):
    workers = [] if worker_pool is None else worker_pool._children
    wake = threading.Event() if worker_pool is None else worker_pool._wake
    force = True
    try:
        while coordinator.has_work or any(worker.job is not None for worker in workers):
            wake.clear()
            if reason := limits.stop_reason():
                jobs = [worker.job for worker in workers if worker.job is not None]
                stop_workers(workers, force=True)
                workers.clear()
                coordinator.cancel(jobs, reason)
                return reason
            for worker in list(workers):
                if worker.job is None:
                    continue
                if limits.stop_reason():
                    break
                timeout = limits.budget.evaluation_timeout_seconds
                inspected = time.perf_counter()
                response = worker.poll()
                while response is not None and response[1] != run_id:
                    # Independent journals reuse attempt IDs, so their results
                    # can only be identified using this lease's unique token.
                    response = worker.poll()
                received = response[4] if response is not None else inspected
                if timeout is not None and received - worker.dispatched >= timeout:
                    workers.remove(worker)
                    stop_workers([worker], force=True)
                    coordinator.timeout(worker.job)
                    continue
                if response is not None:
                    kind, _, attempt, payload, _ = response
                    if kind == "fatal":
                        raise RuntimeError(f"Worker transport/objective failure: {payload}")
                    if kind == "result":
                        if attempt != worker.job[2]:
                            raise RuntimeError("Worker returned a mismatched attempt")
                        coordinator.accept(worker.job, payload)
                        worker.job = None
                        worker.dispatched = None
                        continue
                if response is not None or worker.process.exitcode is not None:
                    workers.remove(worker)
                    stop_workers([worker], force=True)
                    coordinator.retry([worker.job], "worker exited without a result")
            if limits.stop_reason():
                continue
            _retire_idle(workers)
            while not limits.admission_closed:
                if limits.stop_reason():
                    break
                if sum(worker.job is not None for worker in workers) >= coordinator.study.workers:
                    break
                idle = next((worker for worker in workers if worker.job is None), None)
                dispatched = time.perf_counter()
                job = coordinator.dispatch()
                if job is None:
                    break
                limits.admitted += 1
                if idle is None:
                    if worker_pool is None:
                        idle = Worker(objective, wake)
                    else:
                        idle = Worker(None, wake, threads_per_worker=worker_pool.threads_per_worker)
                    workers.append(idle)
                # Startup counts toward the dispatch timeout and wall allowance.
                idle.job, idle.dispatched = job, dispatched
                if worker_pool is not None:
                    worker_pool._run_pids.add(idle.process.pid)
                if limits.stop_reason():
                    break
                idle.submit(job, dispatched, run_id=run_id, objective=objective)
            if not any(worker.job is not None for worker in workers):
                force = False
                return "attempt_budget" if coordinator.has_work else "completed"
            wake.wait(timeout=0.02)
        force = False
        return "completed"
    finally:
        if worker_pool is None or force:
            stop_workers(workers, force=force)
            workers.clear()
        else:
            for worker in workers:
                worker.release(run_id)
