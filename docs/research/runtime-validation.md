# Runtime MVP: validation and boundaries

The harness modules follow the agreed interfaces in `docs/tasks/automl-mvp/CONTRACTS.md`. `Budget` combines saved elapsed time with a monotonic clock and leaves the configured final reserve unavailable to research. `run_process` starts argv directly in a fresh process group, reads stdout and stderr concurrently under one byte cap, samples process-tree RSS with `psutil`, and kills the owned group on timeout, output overflow, memory overflow, or cancellation. Children receive a small environment with explicit host-provided overrides and thread limits; credential-named variables are filtered. `PYTHONPATH` is passed only when the host supplies it.

The provider tests use a local ephemeral HTTP server. They do not call a live model or external API. OpenAI-compatible requests target `/chat/completions` with parallel tool calls disabled; native Ollama requests target `/api/chat` with `stream=false`, `think=false`, and `options.num_predict`. Both adapters return canonical OpenAI-style tool calls with JSON-string arguments and normalized usage. Retries are finite and limited to timeouts, transport failures, rate limits, and server errors; the enclosing timeout includes backoff.

`EventJournal` serializes appends, redacts supplied secret strings and credential fields, assigns sequential records, and fsyncs each append. It truncates an invalid, unterminated final frame only. A malformed newline-terminated frame, sequence gap, or invalid record shape raises `ValueError`. Atomic JSON snapshots use a same-directory temporary file, fsync, replace, and directory fsync.

The subprocess supervisor is resource management, not a security sandbox. It provides no filesystem or network isolation, CPU quota, or disk quota. RSS is sampled, so a very short peak can fall between samples. Process-group cleanup covers descendants that remain in the group; a child that deliberately escapes into another group is outside that guarantee. The platform/container remains responsible for hard isolation and aggregate resource limits. The validation run below used macOS arm64; Linux uses the same POSIX process-session/group APIs, but was not executed in this worktree.

## Red/green transcript

Tests were added against executable `NotImplementedError` stubs before the implementations. This established behavior failures without treating import errors as a RED result.

```text
RED
$ /Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest -q tests/test_runtime.py tests/test_providers.py tests/test_journal.py
FFFFFFFFFFFFFFFFFF                                                       [100%]
18 failed in 3.56s

Representative failure: Budget(...) raised NotImplementedError from automl_agent/runtime.py.
All 18 tests imported their modules and reached the stubbed behavior.
```

After implementing runtime supervision, provider adapters, and journal persistence, the same command passed:

```text
GREEN
$ /Users/egoraprelsky/Desktop/coding/alfa_hackathon/.venv/bin/python -m pytest -q tests/test_runtime.py tests/test_providers.py tests/test_journal.py
..................                                                       [100%]
18 passed in 4.63s
```

Coverage includes resume-aware monotonic budget and reserve, sanitized child environment and thread counts, bounded stdout/stderr, timeout cleanup of a descendant, process-tree RSS termination, provider request shapes and normalization, retry classification and total timeout, secret redaction, torn-tail recovery, mid-file corruption rejection, and atomic JSON replacement.
