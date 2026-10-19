# Audit projection validation

`AuditJournal` keeps `events.jsonl` authoritative and reconstructs `agent.log`
and `decisions.jsonl` from its already-redacted events. The projections use
allowlisted event fields, omit provider message bodies, tool arguments, source
code, raw errors, and unknown event payloads, and retain the original sequence,
timestamp, event name, explicit source (`LLM`, `harness`, or `worker`), and
projected details. Reopening a journal rebuilds projections without replaying
events to stderr. Append-time projection or stderr failures produce a warning
and leave the canonical event durable for later recovery.

The tests were written before the implementation. The initial red run imported
the test stub successfully and failed on all five behavioral checks. After
implementation, all five focused tests passed. The complete repository suite
also passed.

Because each projection is atomically replaced from the canonical journal,
follow a live human log with `tail -F agent.log` so the tail command reopens it
after replacement. Error rows expose a short error code plus the canonical
event sequence and trial `process.json` pointer; full diagnostics remain in the
canonical event/process artifacts. Decision rows label observations,
rationale, alternatives, expected effects, risks, and prior-result assessment.
Assessment rows separately label measured validation metrics and the actual
estimator chain.

| Run | Result | Captured output |
|---|---:|---|
| Initial focused red | 5 failed | [red log](logs/20261019T200428Z-audit-red.json) |
| Focused green | 5 passed | [green log](logs/20261019T200809Z-audit-green.json) |
| Full suite | 70 passed | [suite log](logs/20261019T200809Z-audit-suite.json) |

Commands used:

```text
/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest tests/test_audit.py -q --tb=short
/Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest -q --tb=short
```
