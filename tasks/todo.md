# Reliability bugs: peer links, agent races, reconnects, scheduling, checkpoints, ring memory, tick

Base: RizzyRoger/SlashCompute main at b357ee3. Branch: fix/reliability-bugs. Baseline: 472 passed
(non-integration). Plan: `/Users/darren/.claude/plans/solve-these-bugs-transient-lamport.md`.

- [x] 1. Peer links: keepalive, recv/send/connect timeouts, coordinator stall watchdog
- [x] 2. Peer links survive a dropped connection (seq/ack/retransmit); schedule drops stale/duplicate frames
- [x] 3. Verification ring in host memory, released with the stage
- [x] 4. Agent daemon: one stage handle under a lock, stale commands ignored, atomic status file
- [x] 5. Control messages sequenced, acknowledged and replayed across a reconnect within a grace window
- [x] 6. Transactional epoch start with rollback
- [x] 7. Checkpoint merges serialized per job, atomic, never rewriting a recorded step
- [x] 8. Coordinator: independent loops, WAL, stall-aware heartbeat expiry, blocking work off the loop
- [x] Full suite + integration tests, review the diff, push to SlashComputeFinished

## Results

Full suite: 518 passed, 0 failed (472 at base; 46 new tests). Integration (real coordinator + 2 agents) passes.

- **Stress:** 200-300 in-process runs of a resilient link with three connection cuts each delivered
  every frame exactly once, in order. A 2-stage training run over real TCP with three cuts gives the
  same losses as the in-memory reference.
- **End to end:** a real coordinator and 2 sandboxed agents (the macOS default the integration test
  skips).
  - Freezing one agent daemon (SIGSTOP) past the heartbeat timeout: the coordinator held its place,
    the agent resumed and replayed 206 messages, and the job finished with 0 recoveries.
  - `kill -9` on a worker: reported at once as "worker exited with code -9"; the job recovered from
    its checkpoint once and completed.
- **Found while verifying:**
  - silent drops (no socket close) evicted reliable agents at the heartbeat timeout; they are now
    held like a disconnect
  - a sender retransmitting into a connection the peer had abandoned waited out the whole window;
    links now abort when their read side ends, and bound retransmission at 60 s
  - shutdown could hang when it ran before the agent's Welcome was handled
  - three review findings: spawn failures waited for the 15-minute start timeout, a resume racing a
    close hung LinkServer.close(), and send() succeeded after a link gave up
- **Not verified:** a real multi-Mac run over Wi-Fi, and coordinator restarts (sessions are in
  memory, so a restarted coordinator still restarts epochs from their checkpoints, as before).

# Test every feature and fix bugs (one agent per bug)

Base: RizzyRoger/SlashCompute main at a9758ba (PR #14). Branch: claude/slashcompute-testing-bugs-e28013.

- [x] Baseline suite (the cluster test "failure" was a hidden .pth, see lessons)
- [x] QA agents in parallel: training coordinator/agent, community, web shell + launcher, LLM inference (real llama.cpp b11160), core libs/packaging
- [x] One fixer agent per confirmed bug, each with a regression test; merged here
- [x] Feature: one-time 1 PFLOP welcome credit on sign-in (SLASHCOMPUTE_WELCOME_FLOPS, 0 disables)
- [x] Full suite + real end-to-end inference check
- [x] Working models: Qwen3.5-0.8B-Q4_K_M and Qwen3-8B-Q4_K_M GGUF in ~/models (8B: ~17 tok/s on one M1 Pro)
- [x] PR to RizzyRoger/SlashCompute (critical fixes)
- [x] Minor follow-ups (second PR): CLI 404 traceback + session token, rejected jobs reappearing after restart,
      stale job error, reasoning_content in non-streaming replies/UI, coordinator slow to stop on SIGTERM

## Results (minor pass)

Full suite: 421 passed, 0 failed. Real run with Qwen3.5-0.8B: a non-streaming reply that ran out of tokens while
thinking now returns its reasoning_content (was an empty reply); the coordinator exits 0.16 s after SIGTERM (was >6 s).

## Results (critical pass)

Full suite: 410 passed, 0 failed (base: 277 passed, 1 failed). Real run with Qwen3.5-0.8B split across two
nodes: correct answer, malformed request -> 400 with the pipeline intact, a request right after a mid-stream
disconnect returned in 0 s (was 48 s).

Highest-impact fixes: public-pool nodes could bill/abort/finish other users' jobs and take over node ids; the
local shell's HTTP client reused the last user's session cookie for any caller and had no Origin/Host check;
credit double-spend and settle races; NaN budgets queued free jobs; concurrent signups all became admin; bad
chat requests broke and re-formed the LLM pipeline twice; disconnected streams kept generating; requests
without max_tokens were billed for 256 tokens but generated to the end of the context.

# Clear inference errors, memory controls, and agent/test robustness

Base: GitHub main at 6034d56. Branch: fix/inference-errors-and-memory.

Trigger: uploading a 16.8 GB qwen35 GGUF while joined to a coordinator built before PR #4 failed
with a bare "Not Found"; the GGUF itself parses fine.

- [x] Coordinator capability: launcher reports `inference_supported` from /health; the shell refuses
      model upload and chat with a clear "this coordinator predates LLM inference" error (409)
      before streaming any bytes; the LLMs tab explains it instead of "No models yet".
- [x] Inference node: while it cannot register, write status.json (connecting / unsupported +
      reason) instead of leaving a stale "available" from an older run; 404 on /inference/ping
      logs "coordinator has no LLM inference (update it)".
- [x] Training agent: reconnect with backoff when the coordinator is down or restarts, instead
      of exiting with a traceback (seen at 13:23 and 13:25 in agent.log).
- [x] Memory controls: report this Mac's RAM; training gets a `memory_gb` setting (0 = auto)
      passed as --max-memory-gb and honoured up to the Metal working set; both training and LLM
      memory become sliders sized to the Mac with an "Auto (N GB)" stop.
- [x] Tests: inference harness and late-agent test isolate status/agent files (they wrote to the
      real ~/.slashcompute/inference/status.json and read the real agent status).
- [x] Regression tests for each item; full suite; compare with the known base failure
      tests/test_pipeline.py::test_pipeline_matches_single_stage_reference.
- [x] Review the diff.

## Results

Full suite: 277 passed, 1 failed in 69.79 s. The failure is the known training-loss assertion at
tests/test_pipeline.py:56 with identical numbers on main (6.274 vs 6.088). 9 new tests: unsupported
coordinator (node status, shell 409 with nothing forwarded, launcher capability), agent reconnect after a
1012 restart and stop on a 4003 refusal (both fail on the old daemon), memory setting round trip / argv /
restart-on-change / benchmark honouring the choice, RAM in status. No file under ~/.slashcompute changed
during the run.

Also fixed while verifying in the browser: the LLM model list and pipeline card cached their empty-state
text by data alone, so "Start or join a pool first." stuck after joining; their render keys now include
the state that text depends on.

Checked in a browser against the real pre-inference coordinator at 10.171.167.131 (shell on a temp home):
pill "Pool has no LLMs", explanation in the model list, upload disabled, both memory sliders sized to the
16 GB Mac (max 12 GB) and saving on release.

Not done: the installed /Applications/compute.app bundles its own copy, so these fixes reach it only
after the app is rebuilt.

## Previous: Fix confirmed public-pool and launcher bugs (PR #6, merged)

Full suite at that time: 268 passed, 1 failed (the known pipeline loss assertion, reproduced on main).
