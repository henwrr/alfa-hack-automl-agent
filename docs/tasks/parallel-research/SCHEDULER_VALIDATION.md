# Scheduler validation

The scheduler runs evaluator subprocesses in at most two executor threads. It admits work only while concurrent `n_jobs` stays at or below 8 and the sum of declared memory caps stays at or below 28,000 MB. Completion handling and deadline checks stay on the caller thread; executor threads call only `run_process`.

`run_process` accepts a shared cancellation event, checks it before launch and while polling, and terminates the owned process group when cancellation arrives. The scheduler stops admitting queued work at the research deadline, returns deadline cancellations as `status="cancelled"` with `stderr="research deadline reached"`, and waits for active process-tree cleanup when the generator closes, the callback fails, or `KeyboardInterrupt` occurs. Each process result includes its epoch start/end, PID, and configured `n_jobs` for overlap evidence.

The failure-first check ran `tests/test_scheduler.py` against a scheduler stub: 10 failures showed the missing implementation behavior. After implementation, the focused scheduler suite passed (11 tests), and the full suite passed (101 tests). The overlap case starts two real Python subprocesses with four CPU threads each and confirms intersecting process intervals. Other tests cover independent error/timeout/success results, worker/CPU/memory admission, dynamic deadline cancellation, queue skipping, descendant termination, callback failure, generator close, and `KeyboardInterrupt` cleanup.

Commands and captured output are in [the scheduler validation log](logs/20261019T2035-scheduler.json).
