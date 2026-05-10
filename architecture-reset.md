# Architecture Reset — Implementation Plan

A check-off-able plan for migrating the existing voice agent from "flat
files + LangChain + cloud-by-default" to the architecture in the advisory
panel write-up. Each phase produces a working artifact so the loop never
breaks for more than an evening.

**Order is by risk, low to high.** Don't reorder; later phases assume earlier
phases are in place.

---

## Already in place (from the prior pass)

These were done in the previous session. They count as partial credit toward
the phases below — each phase notes what's still missing.

- [x] Deleted `llm.py` (was dead code)
- [x] Quarantined `flaskserver.py` to `setup/spotify_oauth_bootstrap.py`
- [x] Stripped dead imports from `main.py` and `agent.py`
- [x] Added `config.py` with provider factories (`os.getenv`-based — pydantic-settings still TODO)
- [x] Agent streams tokens via `Agent.stream()`
- [x] Fixed `or → and` in `spotify_client.py:209,229`
- [x] Fixed `msg.type + ":" + msg.content` TypeError
- [x] Replaced hardcoded `thread_id="1"` with `uuid.uuid4()`
- [x] Added `transport.py` with `InProcessTransport`
- [x] Added `privacy.py` with `MicGate` and camera access log
- [x] Async event loop in `main.py` (Edge wraps STT/TTS)

---

## Phase 0 — Repo hygiene (1 hour)

**Goal:** Make the repo reproducible six months from now.

### 0.1 Folder rename
- [x] Rename `robot/` → `robot/` (no spaces, lowercase)
- [x] Update `README.md` paths
- [x] Update `architecture-reset.md` paths (this file)
- [x] Verify `python main.py` still runs from the renamed folder

### 0.2 Lockfile + pinned deps
- [x] Install `uv` (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [x] Run `uv init` inside `robot/` to create `pyproject.toml`
- [x] Move every package from `requirements.txt` into `pyproject.toml` `[project] dependencies`
- [x] Run `uv lock` to produce `uv.lock`
- [x] Delete `requirements.txt` (or keep as a stub pointing to pyproject)
- [x] Verify a fresh `uv sync` followed by `uv run python main.py` works

### 0.3 .env scaffolding
- [x] Create `.env.example` with every variable referenced by `config.py` and `privacy.py`, all values placeholders
- [x] Add comments next to each variable explaining what it does
- [x] Verify `.env` is in `.gitignore` (it is)
- [x] Add a one-line README pointer: "copy .env.example to .env and fill in"

### 0.4 .gitignore + log cleanup
- [x] Add `*.log`, `__pycache__/`, `.DS_Store`, `.venv/`, `models/` to `.gitignore`
- [x] `git rm` the 1.2MB `realtimesst.log` and the `.DS_Store` files in the repo
- [x] Confirm git status is clean of build artifacts

### 0.5 justfile
- [x] Install `just` (`brew install just`)
- [x] Create `justfile` at repo root with at minimum:
  - [x] `just run` — activates venv, runs main.py
  - [x] `just oauth` — runs setup/spotify_oauth_bootstrap.py
  - [x] `just sync` — `uv sync`
  - [x] `just test` — `uv run pytest`
- [x] Verify each `just` target works

**Done when:** `git clone <repo> && cd robot && uv sync && cp .env.example .env && just run` works (assuming .env is filled in).

---

## Phase 1 — Local TTS (1 evening)

**Goal:** Stop paying OpenAI for every spoken response. Match the "all
local" goal in the desk-robot-plan.

### 1.1 Pick a Piper voice
- [x] Browse https://huggingface.co/rhasspy/piper-voices
- [x] Download one English voice (.onnx + .onnx.json) — `en_US-amy-medium` is a reasonable default
- [x] Place under `models/tts/<voice-name>/` (path is gitignored)

### 1.2 Wire Piper into the existing TTS abstraction
- [x] Add `piper-tts` (or use RealtimeTTS' `PiperEngine`) to `pyproject.toml`
- [x] Update `config.make_tts_engine()` so `TTS_PROVIDER=piper` actually loads Piper with the model path from a new `PIPER_VOICE_PATH` env var
- [x] Set `TTS_PROVIDER=piper` in `.env` and `.env.example`
- [x] Run `just run`, say "hello" → verify voice comes from Piper

### 1.3 Measure latency
- [x] Add a print/log of `time.perf_counter()` deltas around: STT-finish, brain-first-token, TTS-first-audio (see `latency.py`)
- [x] Note numbers in this file under "Measurements" below
- [ ] Confirm first-audio latency is under ~500ms (Piper on M1 should be sub-100ms) — currently 1331ms, dominated by stt→token (1249ms = OpenAI cloud round-trip). Phase 2 (local Ollama) and Phase 5 (chunker) should bring this down

### 1.4 STT accuracy + follow-up mode (the "for now" pass)
The wake-word-every-time UX is annoying and the tiny Whisper variant mishears
slurred speech. Two cheap wins before moving on to Phase 2:

- [x] Bump the Whisper model to `medium.en` or `distil-large-v3` in `config.make_stt_recorder()` — one-line change, big accuracy win, still fully local. Note download size in `.env.example` comments (now `STT_MODEL` env var, default `small.en`)
- [x] Add follow-up mode to `Edge`: after the agent finishes speaking, keep the mic open (gated by `MicGate`) for ~15-30s of additional turns without requiring the wake word. After timeout/silence, fall back to wake-word-required (uses `recorder.wakeup()` to bypass wake-word gate per turn, `recorder.abort()` on timeout)
- [x] Document the follow-up window as a `FOLLOWUP_WINDOW_SECONDS` env var
- [x] Privacy invariant: follow-up mode never bypasses `MicGate`. `_listen_followup` returns `None` immediately if the gate is closed

**Done when:** Default TTS produces audio with no network hop. STT mishears less. Talking to the robot feels like a conversation, not a series of commands.

---

## Phase 2 — Provider-agnostic LLM client + drop langchain-openai (1 evening)

**Goal:** Replace `ChatOpenAI` with the `openai` SDK pointed at any
OpenAI-compatible base URL (Ollama, Together, vLLM, OpenAI itself).
One client, many providers. Phase 4's `langchain-openai` removal is bundled
in here since it falls out for free once we stop using `ChatOpenAI`.

### 2.1 Direct OpenAI SDK usage
- [ ] Add `openai` to `pyproject.toml` (already a transitive dep — make it explicit)
- [ ] Write a small `brain/openai_compat.py` that:
  - [ ] Wraps `openai.AsyncOpenAI(base_url=..., api_key=...)`
  - [ ] Exposes a `stream(messages, tools) -> AsyncIterator[token | tool_call]` method matching the Brain protocol
- [ ] Add a `BRAIN_BASE_URL` env var (defaults to OpenAI's URL)
- [ ] Update `config.make_llm()` to return this client when `LLM_PROVIDER=openai-compat`

### 2.2 Hook the new client into the agent
- [ ] LangGraph still wraps the brain — keep the StateGraph + ToolNode for now
- [ ] Bridge: write a thin LangChain-compatible adapter around the new client (or use LangGraph's native message handling) so the existing graph still works
- [ ] Run `just run` against OpenAI — verify nothing regressed
- [ ] Set `BRAIN_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen2.5:32b`, run Ollama locally — verify it talks to Ollama

### 2.3 Strip langchain-openai (was Phase 4, now bundled here)
- [ ] Confirm no remaining import of `langchain_openai`
- [ ] Remove `langchain-openai` from `pyproject.toml`
- [ ] Re-run `uv lock` and `just run` — should still work

**Done when:** Flipping `BRAIN_BASE_URL` is the only change needed to swap between OpenAI and Ollama. No `langchain_openai` in the dep tree.

**Note on local-vs-hybrid:** The goal is local LLM (cost, no metering anxiety, no rate limits, dad's M5 Mini is coming). But no rush. Stay on OpenAI as the default `BRAIN_BASE_URL` until local Qwen runs reliably with every tool you've built. The point of this phase is making the swap a one-env-var change — not making the swap right now.

---

## Phase 3 — `src/` layout + package rename (1 weekend)

**Goal:** Real Python package. Imports work the same in scripts, tests,
and prod. No more "the namespace is the filesystem."

### 3.1 Move files into `src/robot/`
- [ ] Create `src/robot/__init__.py`
- [ ] Move every runtime module under `src/robot/`:
  - [ ] `main.py` → `src/robot/main.py`
  - [ ] `agent.py` → `src/robot/agent.py`
  - [ ] `config.py` → `src/robot/config.py`
  - [ ] `transport.py` → `src/robot/transport.py`
  - [ ] `privacy.py` → `src/robot/privacy.py`
  - [ ] `tts.py`, `stt.py`, `spotify_client.py`, `tool_manager.py`
  - [ ] `tools/` → `src/robot/tools/`
- [ ] Update imports to use the package path (`from robot.config import make_llm`)
- [ ] Configure `pyproject.toml` `[tool.hatch.build.targets.wheel]` (or equivalent) to point at `src/robot`
- [ ] `uv pip install -e .` — verify importable as `import robot`

### 3.2 Subpackage layout
- [ ] Split into the panel's recommended structure:
  - [ ] `src/robot/core/` — placeholder for events, bus, conductor, chunker (Phase 5)
  - [ ] `src/robot/ear/` — move STT here, define `ear/base.py` Protocol
  - [ ] `src/robot/voice/` — move TTS here, define `voice/base.py` Protocol
  - [ ] `src/robot/brain/` — move agent + openai_compat here, define `brain/base.py` Protocol
  - [ ] `src/robot/transport/` — move transport.py here, plus `transport/base.py`
  - [ ] `src/robot/privacy/` — move privacy.py here
  - [ ] `src/robot/tools/inner/` — move spotify + apps tools here
  - [ ] `src/robot/tools/outer/` — empty placeholder for Pipedream MCP
- [ ] Each subpackage has an `__init__.py` re-exporting the public surface

### 3.3 Persona as a file
- [ ] Create `personas/nemo.md` at repo root
- [ ] Move the system-prompt string from `agent.py:14-39` into it
- [ ] `agent.py` reads `personas/nemo.md` at startup
- [ ] `PERSONA_PATH` env var in `config.py` so swapping persona is config

### 3.4 Models directory
- [ ] Move `custom_wakewords/nemo.onnx` to `models/wake/nemo.onnx`
- [ ] Update STT config path
- [ ] Confirm `models/` is gitignored (large binaries don't belong in git)

**Done when:** `uv run python -c "from robot.agent import Agent; print(Agent)"` works from anywhere in the repo. The flat-files era is over.

---

## Phase 4 — Drop the rest of LangChain (DEFERRED — Office Hours, 2026-05-10)

**Status:** The `langchain-openai` removal is bundled into Phase 2.3. The rest of this phase is deferred indefinitely.

**Why deferred:** LangGraph requires `langchain-core` — you can't drop that while keeping LangGraph, and LangGraph is the part you actually want. Removing `langchain` and `langchain-core` is cleanup with no user-visible win, and your time is better spent on capability work (chunker, vision tool, persona) before September 1.

**When to revisit:** When LangChain breaks something you care about, OR when you decide to drop LangGraph itself in favor of a hand-rolled state machine. Until one of those happens, leave it alone.

<details>
<summary>Original Phase 4 plan (struck through — kept for reference)</summary>

> **Goal:** ~~Slim the dependency tree. Keep `langgraph`; lose `langchain-core`, `langchain`, anything else from that family except what LangGraph itself transitively requires.~~
>
> ### ~~4.1 Audit~~
> - [ ] ~~`grep -r "from langchain" src/` — list every import~~
> - [ ] ~~For each, decide: replace with `langgraph`-native, the `openai` SDK, or a plain dict/dataclass~~
>
> ### ~~4.2 Replace~~
> - [ ] ~~Replace `langchain_core.messages.{HumanMessage, SystemMessage}` with LangGraph's message types or plain dicts~~
> - [ ] ~~Replace `langchain_core.tools.StructuredTool` with LangGraph's tool decorator (or pydantic schemas + a registry)~~
> - [ ] ~~Re-test: `just run` works end-to-end~~
>
> ### ~~4.3 Prune~~
> - [ ] ~~Remove `langchain`, `langchain-core`, `langchain-openai` from `pyproject.toml` (whatever remains)~~
> - [ ] ~~`uv lock` — confirm tree shrinks meaningfully~~
> - [ ] ~~Note before/after dep count in "Measurements" below~~
>
> **Done when:** ~~No `from langchain` imports remain. `uv tree` shows a noticeably smaller graph.~~

</details>

---

## Phase 5 — Compressed async core: chunker + events + bus + heartbeats (1 evening + chunker afternoon)

**Goal:** Get the latency win from the chunker, lay the minimum foundations Phase 8 needs (typed events, simple bus, heartbeats), and skip the rest until you actually need it.

**What was cut:** The full Conductor state machine and the full component refactor onto the bus. The current imperative loop in `main.py` is fine for now — the state machine is mostly useful when 3+ components are racing for control, and you have 2 (Ear, Voice). Refactoring components onto the bus is foundation work for Phase 8, but only the *parts* that cross the network boundary matter — internal calls between Ear↔Voice can stay direct.

### 5.1 Sentence chunker (do this first — biggest user-visible win)
- [ ] `core/chunker.py` — accepts an async iterator of tokens, yields phrase-sized chunks
- [ ] Boundaries: `. ! ? ,` or length threshold (~80 chars, tunable)
- [ ] Pipe `Agent.stream()` → chunker → `TextToSpeech.speak()`
- [ ] Measure first-audio latency before/after — record in "Measurements"

### 5.2 Events
- [ ] `core/events.py` — pydantic models for `WakeDetected`, `TranscriptReady`, `BrainToken`, `BrainToolCall`, `SpeakChunk`, `Error`, `Heartbeat`
- [ ] These are the wire format for Phase 8's WebSocket transport. Keep them serializable.

### 5.3 Simple bus
- [ ] `core/bus.py` — async pub/sub on top of `asyncio.Queue` (single broadcast queue is fine for now; topic-per-queue if needed later)
- [ ] Unit test: publish, multiple subscribers, no deadlocks
- [ ] Don't refactor existing components onto it yet. The bus is here so Phase 8's WebSocket transport has somewhere to plug into.

### 5.4 Heartbeats (was Phase 6.4 — moved here so Phase 8 has the foundation)
- [ ] Each long-running component publishes `Heartbeat` events every N seconds
- [ ] Lays the groundwork for Phase 8's reconnect logic — when the WebSocket drops, missing heartbeats are how the conductor will know

### Skipped (was originally Phase 5.2 and 5.4 — Office Hours, 2026-05-10)

Defer until 3+ components race for control (Conductor) or until Phase 8 actually needs the bus (component refactor). Only refactor the components that cross the network boundary.

<details>
<summary>Original Phase 5.2 (Conductor) and 5.4 (component refactor) — struck through</summary>

> ### ~~5.2 Conductor~~
> - [ ] ~~`core/conductor.py` — explicit state machine: IDLE → LISTEN → THINK → SPEAK → IDLE~~
> - [ ] ~~Conductor subscribes to `TranscriptReady`, publishes `BrainToken` and `SpeakChunk`~~
> - [ ] ~~Replace the imperative loop in `main.py` with a Conductor instance plus the components~~
>
> ### ~~5.4 Component refactor onto the bus~~
> - [ ] ~~`Ear` publishes `WakeDetected` and `TranscriptReady` instead of returning text~~
> - [ ] ~~`Voice` subscribes to `SpeakChunk` instead of being called~~
> - [ ] ~~`Brain` driven by Conductor, publishes `BrainToken`/`BrainToolCall`~~
> - [ ] ~~Manual e2e test — wake, speak, response~~

</details>

**Done when:** Chunker is in. First-audio latency is meaningfully lower (target: <500ms once you also swap to local LLM). Events + bus + heartbeats exist as scaffolding for Phase 8.

---

## Phase 6 — Persistence + privacy gate + logging (1 evening)

**Goal:** Survive restart. Survive Wi-Fi blips. Tell the operator what's happening.

### 6.1 SqliteSaver memory
- [ ] Replace `MemorySaver` in `agent.py` with `SqliteSaver` pointed at `state/conversations.db`
- [ ] `state/` is gitignored
- [ ] Decide on `thread_id` policy: per-day, per-wake-event, or per-explicit-session — document in this file
- [ ] Add `forget_session` and `recall(query)` tools so cross-session memory is reachable from the agent

### 6.2 Privacy gate, hardened
- [ ] Move `MicGate` into `privacy/gate.py` if not already
- [ ] All audio reads from `Ear` go through the gate — gate-closed = silent no-op, no buffering
- [ ] Hard cap of `MAX_UTTERANCE_SECONDS` enforced on Ear's side, not just trusted from the recorder
- [ ] Toggle hook for a future hardware mute pin (placeholder function)
- [ ] Camera access log already exists — confirm every code path that opens the camera calls `log_camera_access()`

### 6.3 Structured logging
- [ ] Add `loguru` (simple) or `structlog` (structured JSON)
- [ ] Replace every `print()` in `src/robot/` with a logger call
- [ ] Set log level via env var (`LOG_LEVEL=INFO`)
- [ ] Verify log output is grep-friendly

### 6.4 Health checks (Heartbeat publishing moved to Phase 5.4 — Office Hours, 2026-05-10)
- [ ] Watchdog task in `main.py` subscribes to `Heartbeat` events from the bus; logs a warning if any component goes silent for >N seconds. (No formal Conductor yet — simple async task is enough.)
- [ ] Lays groundwork for Phase 8's WebSocket reconnect logic.

<details>
<summary>Original 6.4 task — struck through (heartbeats now live in Phase 5.4)</summary>

> - [ ] ~~Heartbeat event published every N seconds by each long-running component~~

</details>

**Done when:** Killing and restarting the process picks up the conversation. Toggling `MIC_ENABLED=false` immediately silences the mic. Logs are JSON-parseable.

---

## Phase 7 — Tests (1 evening, ongoing)

**Goal:** Catch the obvious regressions. Not 100% coverage; the smallest
set that makes future changes safe.

### 7.1 pytest scaffold
- [ ] Add `pytest` and `pytest-asyncio` to dev deps
- [ ] `tests/` directory at repo root
- [ ] `just test` runs `uv run pytest`

### 7.2 Unit tests for the easy wins
- [ ] `tests/test_chunker.py` — sentence boundaries, length threshold, partial-token handling
- [ ] `tests/test_bus.py` — pub/sub, fan-out, no leaks
- [ ] `tests/test_spotify_client.py` — mock `requests`, verify the `200 <= status < 300` fix doesn't regress
- [ ] `tests/test_privacy_gate.py` — gate-open vs. gate-closed, cap enforcement

### 7.3 Smoke test
- [ ] One end-to-end test that spins up the conductor with stub Ear/Voice/Brain and verifies a TranscriptReady → SpeakChunk round trip

**Done when:** `just test` passes in CI-like fresh-clone conditions.

---

## Phase 8 — Pi/Mac split (the big one)

**Goal:** Run Edge on a Pi, Brain on the Mac, talking over WebSocket.
This is what the whole architecture has been pointing at.

### 8.1 WebSocket transport
- [ ] `transport/websocket.py` — implements the same `Transport` protocol as `InProcessTransport`
- [ ] Server side (Mac): receives events, dispatches to Brain, streams tokens back
- [ ] Client side (Pi): wraps Ear/Voice, forwards events
- [ ] Reconnect with exponential backoff
- [ ] Auth: shared-secret token in `.env`

### 8.2 Cross-machine smoke test
- [ ] Run server on Mac, "client" still on Mac (loopback) — verify works
- [ ] Run client on a second Mac/laptop over Wi-Fi — verify works
- [ ] Latency budget check: total round-trip should still be under 2s perceived

### 8.3 Pi hardware
- [ ] (Per desk-robot-plan Phase 3) buy hardware after Phase 1 of that plan passes
- [ ] Flash Pi, install client deps, deploy `src/robot/` (Edge subset only)
- [ ] systemd unit so it auto-starts on boot
- [ ] Wi-Fi reconnect tested by toggling the router

#### 8.3.a Hardware path — decide before buying
Two realistic options. Default to (A) unless the far-field mic array on the
Echo turns out to actually matter for the desk environment.

- **(A) Off-the-shelf — recommended default.** Pi 4/5 + USB mic + small USB
  or 3.5mm speaker + 4 GPIO buttons in a printed/cut case. ~$40-50 total
  beyond the Pi. ~1 afternoon of assembly. Easy to debug, easy to replace
  parts.
- **(B) Mod the Echo Dot.** Have a used Echo Dot (sphere with flat base, 4
  buttons, never set up). Gut Amazon's mainboard, keep the shell, mic
  array, speaker, button board, LED ring; drive everything from a Pi
  inside.
  - **Why bother:** Echo's 7-mic far-field array is genuinely good in
    noisy rooms; the case + buttons + speaker save ~$40-50; aesthetic
  - **Why not:** the mics are usually PDM, which the Pi can't read
    natively — needs a ReSpeaker HAT or PDM→I2S converter (~$15-30),
    which eats most of the savings. Pinouts vary by Echo generation and
    aren't documented. Real soldering, real risk of bricking. ~2-3
    weekends if hardware isn't your usual thing
  - **Pre-decision homework before committing:**
    - [ ] Identify the exact Echo Dot generation (model number is on the
      base sticker — gen 2/3 is plastic + screws, gen 4/5 is the sphere
      and uses clips)
    - [ ] Find a teardown/pinout writeup for that generation (forums,
      iFixit, YouTube). If nothing exists, mod cost balloons
    - [ ] Confirm mic interface (PDM vs. analog vs. I2S) and price the
      adapter board the Pi needs
    - [ ] Re-run the savings math after adapter cost — if net savings is
      under ~$20, just buy the off-the-shelf parts
  - **Decision rule:** only pick (B) if (1) a teardown exists for this
    exact generation, AND (2) far-field pickup actually matters for the
    use case, AND (3) the hardware-project time is something you'd enjoy
    rather than tolerate

### 8.4 Lockfile portability — REVISIT before Pi work
- [ ] Widen `[tool.uv] environments` in `robot/pyproject.toml` to include `sys_platform == 'linux' and platform_machine == 'aarch64'`
- [ ] Currently pinned to Mac arm64 only because `openwakeword>=0.6` pulls `tflite-runtime` on Linux, which has no `cp312` wheel
- [ ] Likely fix: split deps so the Pi (Edge) installs `openwakeword` + `RealtimeSTT` + `pyaudio`, and the Mac (Brain) installs `langchain*` + `openai`. Each side gets a smaller dep tree and a portable lockfile
- [ ] Same goes for `pyaudio` — the `no-binary-package` workaround is Mac-specific (Homebrew portaudio); on Linux the wheel works fine

**Done when:** Pi sitting on the desk, Mac in the closet, full conversation works.

---

## Future considerations

Ideas worth keeping but not scheduled yet. Revisit when one of them feels
like the obvious next move.

### STT model upgrades
Whisper variants, all local, all loadable via the same RealtimeSTT/`faster-whisper` path. Tradeoff is size vs. accuracy vs. speed:
- `base` (~140MB) — easy free win over `tiny`
- `small.en` (~450MB) — sweet spot for most desk-robot use
- `medium.en` (~1.5GB) — near-human English; covered in 1.4
- `distil-large-v3` (~750MB) — ~95% of large-v3 quality at 3x the speed; covered in 1.4
- `large-v3` (~3GB) — overkill unless dictation
- **Non-Whisper:** NVIDIA Parakeet (RNN-T, ~600MB, very fast on English) — needs a non-RealtimeSTT path
- **WhisperX** — same accuracy, better VAD + word-level timestamps; more robust to trailing silence

### Wake-word UX alternatives
The "say nemo every time" UX gets old. Options, roughly ordered by lift:
- **Follow-up mode** — wake once, mic stays open for N seconds of follow-ups. Covered in 1.4. Smallest lift, biggest UX win, preserves privacy default
- **On/off voice toggle** — "hey nemo, listen up" / "okay nemo, stop". Mic stays open across the entire on-period. Privacy regression vs. follow-up mode; only worth it for hands-busy flows (cooking, etc.)
- **Hardware toggle** — physical switch or key chord. Most reliable, no false wakes, lands naturally on the Pi (Phase 8) where a hardware mute pin is already on the roadmap
- **VAD-driven turn detection** — no wake word needed mid-conversation; just detect "user starts speaking → user stops → that's a turn". Wake word only needed to start a session OR after a long idle window. Closest to natural conversation. Bigger lift but compounds well with follow-up mode
- **Different wake phrase** — longer phrases like "hey nemo" or "computer" false-trigger less often; only worth it if false-fires are the actual annoyance, not the wake-every-time friction
- **Retrain/replace the "nemo" model** — current custom `nemo.onnx` triggers inconsistently. Either retrain with more samples (openWakeWord training notebook) or swap to a built-in openWakeWord phrase ("hey jarvis", "alexa", "computer") that's better-trained out of the box

---

## Measurements

Fill in as we go. The numbers tell us whether we're winning.

| Metric | Before | After Phase 1 (Piper) | After Phase 5 (chunker) | Target |
|---|---|---|---|---|
| First-audio latency | _measure_ | 1331ms total (stt→token 1249ms, token→audio 82ms) | | <500ms |
| End-to-end response time | _measure_ | | | <2s perceived |
| `requirements.txt` line count | ~40 | | | |
| Direct `langchain*` deps | 3 | | | 0 |
| Total deps in `uv.lock` | _measure_ | | | meaningfully smaller |

---

## Decisions log

When a phase forces a non-obvious choice, write a one-line entry here so
future-you doesn't re-litigate it.

- _e.g._ "Phase 6: thread_id is per-day rather than per-wake — preserves cross-conversation continuity within a day, resets if the robot has been idle overnight"

### From Office Hours review (2026-05-10)

- **Local LLM is the goal, not a v1 blocker.** Stay on OpenAI/Anthropic via Phase 2's provider-agnostic client. Swap to local Qwen on dad's M5 Mini (when it arrives) by changing one env var. Reasons: no metering anxiety, no rate limits, the M5 Mini will sit there capable. API cost itself was not the real driver — at hobby usage it's a few dollars a year.
- **Phase 4 standalone is dead.** `langchain-openai` removal moves into Phase 2.3. The rest is cleanup with no user-visible win and is deferred indefinitely. LangGraph requires `langchain-core` so dropping it is impossible while keeping LangGraph.
- **Phase 5 is compressed.** Chunker (5.1) + events (5.2) + simple bus (5.3) + heartbeats (5.4). Skip the full Conductor state machine and the full component refactor onto the bus. They're foundation work for a problem you don't yet have (3+ components racing for control). Revisit when Phase 8 actually needs the bus, and only refactor the components that cross the network boundary.
- **Pi/Mac split stays.** Confirmed for next several months: Pi on the desk, brain on the laptop. When dad's M5 Mini arrives, brain moves there with one env var change (`BRAIN_BASE_URL`). Wake-on-LAN + Tailscale set up in advance so the swap is painless.
- **Mac sleeping is a non-issue.** Laptop is always on at the desk. `caffeinate` deferred until it ever annoys you.
- **v1 ship date: September 1, 2026.** See desk-robot-plan.md §1.7 for the polish-list cutoff.
