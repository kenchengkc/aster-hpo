"""Owned spawn workers with isolated channels and bounded group shutdown.

Parent exchange threads handle blocking pickle/pipe I/O; the coordinator polls
only local mailboxes. A terminated worker's channel is never reused.
"""

import multiprocessing
import queue
import signal
import threading
import time

from .core import _evaluate


def _worker_loop(objective, connection):
    # The coordinator handles Ctrl-C and controls worker cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while (request := connection.recv()) is not None:
            attempt, config, replication = request
            try:
                observation = _evaluate(objective, config, replication)
            except BaseException as exc:
                connection.send(("fatal", attempt, f"{type(exc).__name__}: {exc}"))
                return
            connection.send(("result", attempt, observation))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class Worker:
    def __init__(self, objective, wake):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_worker_loop, args=(objective, child), name="aster-worker"
        )
        self.job = None
        self.dispatched = None
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
        try:
            while True:
                request = self.requests.get()
                self.connection.send(request)
                if request is None:
                    return
                self.responses.put(self.connection.recv())
                self.wake.set()
        except (EOFError, OSError) as exc:
            self.responses.put(("lost", None, f"{type(exc).__name__}: {exc}"))
            self.wake.set()
        except BaseException as exc:
            self.responses.put(("fatal", None, f"{type(exc).__name__}: {exc}"))
            self.wake.set()
        finally:
            self.connection.close()

    def submit(self, job, dispatched):
        self.job, self.dispatched = job, dispatched
        trial, replication, attempt = job
        self.requests.put((attempt, trial.config, replication))

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


def run_parallel(coordinator, objective, limits):
    workers = []
    wake = threading.Event()
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
                if timeout is not None and time.perf_counter() - worker.dispatched >= timeout:
                    workers.remove(worker)
                    stop_workers([worker], force=True)
                    coordinator.timeout(worker.job)
                    continue
                response = worker.poll()
                if response is not None:
                    kind, attempt, payload = response
                    if kind == "fatal":
                        raise RuntimeError(f"Worker transport/objective failure: {payload}")
                    if kind == "result":
                        if attempt != worker.job[2]:
                            raise RuntimeError("Worker returned a mismatched attempt")
                        coordinator.accept(worker.job, payload)
                        worker.job = None
                        continue
                if response is not None or worker.process.exitcode is not None:
                    workers.remove(worker)
                    stop_workers([worker], force=True)
                    coordinator.retry([worker.job], "worker exited without a result")
            if limits.stop_reason():
                continue
            while not limits.admission_closed:
                if limits.stop_reason():
                    break
                idle = next((worker for worker in workers if worker.job is None), None)
                if idle is None and len(workers) >= coordinator.study.workers:
                    break
                dispatched = time.perf_counter()
                job = coordinator.dispatch()
                if job is None:
                    break
                limits.admitted += 1
                if idle is None:
                    idle = Worker(objective, wake)
                    workers.append(idle)
                # Startup counts toward the dispatch timeout and wall allowance.
                idle.job, idle.dispatched = job, dispatched
                if limits.stop_reason():
                    break
                idle.submit(job, dispatched)
            if not any(worker.job is not None for worker in workers):
                force = False
                return "attempt_budget" if coordinator.has_work else "completed"
            wake.wait(timeout=0.02)
        force = False
        return "completed"
    finally:
        stop_workers(workers, force=force)
