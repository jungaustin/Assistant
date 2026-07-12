# Desk Robot — Project Plan

A palm-sized, voice-driven desk companion that extends Austin's existing music-playing agent. Runs as a thin client streaming audio + video over Wi-Fi to a brain hosted on a Mac (M1 Max MacBook Pro now, dedicated Mac Mini later).

---

## 1. Vision & Goals

**What it is:** A small, always-on (when the Mac is on) physical robot that lives on the desk. You talk to it, it talks back, it can see what's in front of it, and it can take actions on your behalf — play music, send messages, eventually more.

**Success criteria — v1 is "done" when:**
- You can say a wake word, ask it to play a song or send a message, and it does it reliably.
- It responds with natural-sounding voice in a conversational latency window (<2s perceived).
- It can answer "what do you see?" using its camera.
- It runs on a real piece of hardware sitting on the desk, not just on the laptop screen.
- All inference is local (LLM on the Mac, STT/TTS/wake-word on the robot or Mac).

**Non-goals for v1:** Mobility, arms, screen-based facial expressions, multi-user voice ID, internet-reachable access (home network only).

### 1.5 User persona

Austin, alone, in his room. Single user, single voice, single environment, single network. Every design decision is evaluated against this user — not "users in general," not "future family members," not "what if someone else uses it." If a feature only matters for a hypothetical second user, it's v2 or later.

### 1.6 Failure UX principle

**The robot always tells you what it's doing, even when failing.** Silence is the worst failure mode — the user can't tell whether the robot heard them, is thinking, broke, or is ignoring them.

Sub-rules:

1. **Never silent.** If the robot can't do something, it says so out loud. No silent dropped utterances, no swallowed exceptions.
2. **Audible "thinking" cue if a response takes >2s.** A soft tone or filler word. Silence-during-thinking is the worst feeling.
3. **LED states are an information channel.** Idle, listening, thinking, speaking, error — distinct colors. Promoted from "Future / wouldn't it be cool if" to a v1 requirement.

Concrete implications:

- **Wake word doesn't fire:** LED stays off (already a signal). No audible cue — would trigger on every conversation in the room. User learns to glance at the LED.
- **STT mishears:** Agent confirms before destructive or ambiguous actions ("Play Vito by NewJeans?"). Cheap path: prompt the LLM to confirm song/contact names. Better path (later): tool wrappers do fuzzy matching against known entities and return "did you mean X?" when confidence is low.
- **Tool call fails:** Wrap every tool call in `try/except` at the agent layer. On exception, inject a natural-language error message back into the conversation so the LLM speaks it ("Hmm, that didn't work — try again?"). Never a stack trace. Never silence.

### 1.7 v1 ship date

**Target: September 1, 2026.**

v1 is shipped when the §1 success criteria are met AND the failure UX principle is honored across all known failure paths. Everything else is v2.

**v1 polish — only these 3 items count:**
1. LED status states (required by failure UX principle)
2. One persona pass on the system prompt (1 hour, then stop)
3. Wake-on-LAN + Tailscale setup for the eventual M5 Mini (so the brain swap is one config change)

Every other Phase 7 idea — calendar, weather, smart home, multi-room, voice ID, mood/affect, motion, OLED display — is **explicitly v2.** Don't touch them before September 1.

---

## 2. Architecture

```
┌─────────────────────┐         Wi-Fi          ┌──────────────────────────┐
│   The Robot (Pi)    │  <── audio in/out ──>  │    Mac (M1 Max → Mini)   │
│                     │  <── video frames ──>  │                          │
│ • Mic + speaker     │                        │ • Ollama (Qwen 2.5 32B)  │
│ • Camera            │                        │ • Vision model (Qwen-VL) │
│ • Wake-word detect  │                        │ • Agent runtime          │
│ • Status LED        │                        │ • Tool integrations      │
│ • Streams to Mac    │                        │ • Existing music agent   │
└─────────────────────┘                        └──────────────────────────┘
```

**Why hybrid (brain on Mac, I/O on Pi):**
- Lets you run a 32B-class model with reliable tool calling — much better than anything that fits on a Pi.
- Robot stays palm-sized because no thermal/compute burden.
- Easy to upgrade the brain (swap models, eventually move to dedicated Mini) without touching the robot.
- Each side can be developed and tested independently.

**Communication:** WebSocket over local Wi-Fi. Robot opens a persistent connection to the Mac on boot. Audio streams in both directions; camera frames sent on-demand when the agent decides to "look."

---

## 3. Phased Build Plan

The order matters. Each phase produces a working artifact you can use, and de-risks the next phase. **Don't buy hardware until Phase 3 is done.**

### Phase 0 — Local LLM running on the Mac (1 evening) — ✅ DONE 2026-06-12

> Ollama 0.30.8 running qwen2.5:32b (and 14b pulled for comparison). OpenAI-compatible endpoint verified. `just run` starts/stops the server and pre-warms the model. ~19GB resident for 32b on the M1 Max / 64GB.

**Goal:** Confirm Ollama works, the chosen model loads, and you can chat with it.

- Install Ollama (`brew install ollama` or the .pkg)
- `ollama pull qwen2.5:32b`
- `ollama run qwen2.5:32b` — confirm it responds
- Verify the OpenAI-compatible endpoint at `http://localhost:11434/v1/chat/completions` with a curl test
- Note RAM usage during inference (Activity Monitor) so you know real-world footprint with your normal apps open

**Done when:** You can hit the Ollama API from the terminal and get a response.

---

### Phase 1 — Wire existing music agent to the local model (1-2 evenings) — ✅ DONE 2026-06-12

> Brain swapped from `gpt-4o-mini` to local qwen2.5:32b via env vars only (`BRAIN_BASE_URL`/`LLM_MODEL`), no code change. Tool calling verified reliable across all 22 tools — cleared the plan's #1 risk. Root-caused the early 20s latency to Ollama's default 2048-token context overflowing the tool schemas; fixed with `OLLAMA_CONTEXT_LENGTH` (now 16384). 14b was faster but leaked multilingual gibberish, so 32b is the daily driver.

**Goal:** Prove that local-model tool calling is reliable enough for this project. This is the most important early signal.

- Point your existing music-agent codebase at `http://localhost:11434/v1` instead of whatever it currently uses (likely a one-line config change if it's OpenAI-SDK-based)
- Run your existing tool-call test cases
- Measure reliability: out of 20 attempts at "play [song]", how many succeed cleanly? How many produce malformed tool calls?
- If <90% reliable on simple tools: try `qwen2.5:14b` for speed comparison, or fall back to a hosted API for the brain (Claude/OpenAI) and only run STT/TTS locally
- If reliable: continue with local

**Decision point:** Fully local vs. hybrid (local for I/O, hosted API for the brain). Defer until you see real numbers from this phase.

**Done when:** You can run the music agent against the local model and it reliably executes tool calls.

---

### Phase 2 — Voice loop on the laptop (2-3 evenings) — ✅ DONE

> Full hands-free loop in `main.py`: openWakeWord ("nemo") → RealtimeSTT (`small.en`) → local brain → Piper TTS, with a follow-up window so you don't re-say the wake word, a MicGate privacy switch, and a boot "ready" beep. This is the current daily-run state via `just run`.

**Goal:** Make the agent fully voice-controlled, running entirely on the laptop. The "robot" at this stage is just the laptop with a mic and speakers — you're proving the conversation loop works end-to-end before introducing hardware.

Components:
- **Wake-word detection:** [openWakeWord](https://github.com/dscripka/openWakeWord) (free, local, decent accuracy) or [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) (better accuracy, free for personal use). Pick a custom wake phrase like "hey scout" or whatever you want to call it.
- **Speech-to-text:** [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp) with the `base.en` or `small.en` model. Fast on M1, runs locally.
- **Text-to-speech:** [Piper](https://github.com/rhasspy/piper) for fast natural-sounding TTS, or [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for higher quality. Both local. Try both, pick the voice you like.
- **Glue:** A small Python script that wires it all together: wake word → record audio until silence → transcribe → send to agent → speak the response. The agent itself is your existing repo, now extended with a `speak()` capability.

**Done when:** You can say "hey [name], play some lo-fi" while standing across the room and it works without touching the laptop.

---

### Phase 3 — Vision capability (1-2 evenings) — ⏸️ DEFERRED 2026-06-12

> Deprioritized by user in favor of Tier 1/2 tools first. Not started. Pick back up after the agentic-tools backlog or alongside the Phase 8 hardware work.

**Goal:** Add a "look at the world" tool the agent can call.

- Pull a vision-language model: `ollama pull qwen2.5vl:7b` (lighter, faster) or larger if you want better quality
- Add a tool to the agent: `look(question: str) -> str` that captures a frame from the laptop's webcam, sends it to the VLM with the question, returns the answer
- Test cases: "what am I holding?", "is the door open?", "describe my desk"
- Decide on calling pattern: agent calls vision only when needed (cheaper, more latency) vs. periodic captures (more proactive but heavier)

**Done when:** You can ask "what do you see?" and get a coherent answer from the laptop's webcam.

> **Up to here, you've built the entire robot in software. Everything works on the laptop. The hardware phases below are just about giving it a body.**

---

### Phase 4 — Hardware bench prototype (1 weekend + parts shipping time)

**Goal:** Get a Pi sitting next to your laptop, running the robot client software, talking to the brain. No enclosure yet — wires everywhere, that's fine.

**Recommended hardware (see full shopping list in §5):**
- Raspberry Pi 5 (4GB is plenty since the Pi isn't running models) or Pi Zero 2 W if you want to push smallest possible
- USB conference mic or I²S MEMS mic (better for far-field) — ReSpeaker 2-Mics Pi HAT is the popular pick
- Small speaker — either a USB speaker or an I²S amp + small driver
- Camera — Pi Camera Module 3 (best image quality) or a cheap USB webcam (easier to mount)
- microSD card (32GB+), USB-C power supply, case for testing
- A single addressable RGB LED (WS2812) for status — looks much nicer than a regular LED

**Tasks:**
- Flash Raspberry Pi OS Lite (no desktop), set up SSH and Wi-Fi
- Install audio stack: ALSA + pipewire, confirm mic and speaker work with `arecord`/`aplay`
- Install camera stack: `libcamera`, confirm capture with `libcamera-still`
- Run a "hello world" client: capture audio, send raw bytes to a test WebSocket on the Mac, play received audio back

**Done when:** You can speak into the Pi's mic, the audio reaches the Mac, and the Mac plays a response back through the Pi's speaker.

---

### Phase 5 — Full robot client software (1-2 weekends)

**Goal:** All the pieces working together on the robot, with the brain on the Mac.

- **On the robot (Pi):**
  - Wake-word detection (openWakeWord runs fine on a Pi 5)
  - On wake: start streaming mic audio over WebSocket to the Mac, light up the LED
  - Stream incoming audio from Mac to speaker
  - Camera capture endpoint — when the Mac requests a frame, grab one and send it
  - Reconnect/retry logic for when the Mac is asleep or the network blips
- **On the Mac:**
  - WebSocket server that:
    - Receives streamed audio, runs Whisper
    - Sends transcript to the agent
    - Receives agent text response, runs TTS, streams audio back
    - Handles vision requests by asking the robot for a camera frame
  - Auto-launches on login (use `launchd`, not Login Items, so it survives logouts)
  - Caffeinate-style sleep prevention while serving

**Done when:** You can say the wake word at the robot on your desk and it works, hands-free, no laptop interaction.

---

### Phase 6 — Enclosure & form factor (1-2 weekends, depending on access to a 3D printer)

**Goal:** Make it look and feel like a robot, not a project board.

**Big decisions to make before this phase:**
- **Aesthetic direction:** Minimalist (clean cylinder/cube), retro (Wall-E vibes), creature (eyes, antenna)
- **Display:** None / status LED only? Small OLED for eyes/expression? Round LCD?
- **Movement:** None? A nodding head servo? Full pan/tilt? Decide based on how much you want to fight with mechanical design.
- **Power:** Always plugged in (USB-C from a wall wart) or battery-powered with a charging dock?

**Tasks:**
- Sketch in 2D first, then model in Fusion 360 / OnShape / Blender
- Account for mic placement (matters a lot for far-field), speaker chamber, camera line-of-sight, ventilation, cable routing
- Print at low infill, test fit, iterate
- Soft-mount the components (foam/silicone) to reduce vibration noise into the mic

**Done when:** You can hand it to someone and they say "oh, it's a robot," not "what's that breadboard?"

---

### Phase 7 — Polish, persona, and expansion (ongoing)

Once the core works, this is where it gets fun:

- **Persona:** Give it a name, a system prompt with personality, a consistent voice. This is what makes it feel like *yours*.
- **More tools:** Calendar, weather, smart home, timers, reminders, search, "remember this for me" memory.
- **Better wake interactions:** Visual feedback (LED color states: idle, listening, thinking, speaking, error).
- **Mac Mini server upgrade:** When you get an M4/M5 Mini, move the brain there. Robot doesn't need to know — same Wi-Fi, just a different IP.
- **Optional motion:** Servo for "looking at" the speaker. Easy nice-to-have.
- **Optional display:** Small OLED with animated eyes — simple but huge personality boost.
- **Multi-room:** Build a second robot with the same client code; both connect to the same brain.

---

## 4. Software Stack Summary

| Layer | Component | Where |
|---|---|---|
| Wake word | openWakeWord or Porcupine | Robot |
| Speech-to-text | whisper.cpp (`small.en`) | Mac |
| LLM | Qwen 2.5 32B Instruct via Ollama | Mac |
| Vision | Qwen 2.5 VL 7B via Ollama | Mac |
| Text-to-speech | Piper or Kokoro | Mac |
| Agent runtime | Your existing music-agent repo, extended | Mac |
| Transport | WebSocket over local Wi-Fi | Both |
| Robot OS | Raspberry Pi OS Lite | Robot |
| Robot client | Python (mic/speaker/camera/wake/WS) | Robot |
| Mac service | Python service auto-launched via `launchd` | Mac |

---

## 5. Hardware Shopping List (rough)

**Core (~$100-150):**
- Raspberry Pi 5 (4GB) — ~$60
- microSD 64GB A2 — ~$12
- USB-C 27W power supply — ~$15
- Pi Camera Module 3 — ~$25
- ReSpeaker 2-Mics Pi HAT — ~$15
- Small full-range speaker (3W, 4Ω) + I²S amp — ~$10
- WS2812 RGB LED or small ring — ~$5
- Tactile button (10-12mm momentary, panel-mount) for hardware mic mute — ~$2 — **required for v1** (privacy gate is software; the button cuts mic power at the hardware level)

**Nice-to-have:**
- Small OLED (SSD1306 128x64) for face/status — ~$5
- Micro servo (SG90) if you want any motion — ~$3
- Battery + charging board if you go untethered — ~$25

**Tools you may need:**
- Soldering iron (cheap one is fine)
- 3D printer access (makerspace, library, or a friend) — or order prints from JLC3DP / SendCutSend

---

## 6. Open Decision Points (defer until you reach the relevant phase)

1. **Fully local vs. hybrid brain.** ~~Decide after Phase 1 measurement.~~ **DECIDED:** local is the goal; no rush. Stay on OpenAI/Anthropic via the Phase 2 provider-agnostic client until local Qwen on the M5 Mini is reliable with all tools. Then flip one env var.
2. **Form factor & aesthetic.** Decide before Phase 6.
3. **Display: none / LED only / OLED eyes / LCD face.** Decide before Phase 6. **Note:** LED is required for v1 failure UX (see §1.6).
4. **Movement: none / nodding head / pan-tilt.** Decide before Phase 6.
5. **Power: tethered USB-C vs. battery + dock.** Decide before Phase 6.
6. **Wake word.** Decide before Phase 2.
7. **Voice (TTS): which Piper voice or Kokoro voice.** Decide during Phase 2.
8. **Specific messaging integrations.** Decide as you add them in Phase 7 (iMessage via macOS shortcuts, Discord via webhook, Slack, SMS via Twilio, etc.)
9. **Hardware mic mute switch.** ~~Spec before Phase 6.~~ **DECIDED:** yes, in v1. Tactile button cutting mic power at the hardware level. Spec into Phase 4 hardware list.

---

## 7. Risk Areas (things that will probably bite you)

- **Tool-calling reliability with local models.** This is the #1 risk. Mitigation: Phase 1 explicitly tests this. Hybrid is the fallback.
- **Latency.** The full loop (wake → STT → LLM → TTS → playback) can easily hit 3-5s if you're not careful. Streaming TTS as the LLM generates helps a lot. So does using a smaller model.
- **Mac sleeping.** macOS aggressively sleeps. Use `launchd` + `caffeinate -i -d` while the service is active, and configure power settings to never sleep when on AC. **For v1: non-issue.** Laptop is always on at the desk when the robot is in use. Revisit if the robot ever needs to work while the laptop is closed.
- **Mic quality.** A bad mic kills the experience. Far-field with a single cheap mic is rough. The ReSpeaker 2-Mics HAT or a small mic array makes a huge difference.
- **Speaker quality.** A tinny speaker makes the whole thing feel like a toy. Spend the extra $5 on something with a real enclosure or design a sealed chamber in your 3D print.
- **Wi-Fi reliability.** WebSocket reconnect logic isn't optional. Plan for it from the start.
- **Privacy/security.** This thing has a mic and camera always on. Worth building in a hardware mute switch (physical button that cuts mic power) if that matters to you.

---

## 8. What to do tonight

1. Install Ollama, pull `qwen2.5:32b`, run it (Phase 0 — 30 minutes).
2. Open your existing music agent repo, change the API endpoint to point at Ollama.
3. Run your existing test cases. See what happens.
4. Report back with: how reliable was tool calling? How did latency feel? Did the M1 Max breathe okay during inference?

That's the only experiment that actually tells you whether the rest of the plan is viable as fully-local. Everything else can be planned around the answer.

---

## 9. Agentic Tools Backlog

Tools to build, roughly in priority order. All are Phase 7 scope unless noted.

### Tier 1 — High daily use, easy to ship

| Tool | How | Notes |
|---|---|---|
| **Timer** | `threading.Timer` + beep callback | ✅ Built 2026-06-12. Background timers (keep talking while one runs), set/list/cancel, double-blip on done. No API. |
| **Weather** | ~~`wttr.in`~~ | ⏭️ Skipped 2026-06-12 — the Tavily `web_search` tool answers weather cleanly, so a dedicated tool is redundant. |
| **Web search** | Tavily API | ✅ Built 2026-06-12. Answer-synthesis mode (one spoken paragraph, not raw links). Lazy client; missing key degrades gracefully. |

### Tier 2 — High value, slightly more work

| Tool | How | Notes |
|---|---|---|
| **Discord** | Bot token, REST only | ✅ Built 2026-06-12. Send + catch-up (summarize since last read). Nemo keeps its own per-channel cursor (Discord hides personal read state from bots); `since` override + mark-as-read handle cursor staleness. Awaiting bot creation in the Developer Portal. |
| **Nightly check-in prompt** | Scheduled task at ~22:00 | ✅ Built 2026-06-12 in `core/checkin.py`, wired into `main.py`. Background asyncio task; at `DAILY_LOG_PROMPT_HOUR` asks about unlogged `DAILY_REQUIRED_TYPES`, silent if complete. Nemo's first proactive behavior. |

### Tier 3 — Useful but defer

| Tool | Why defer |
|---|---|
| Smart home (HomeKit/Home Assistant) | Only worth it once the physical robot is on the desk |
| Clipboard read/write | Niche; typing is faster |
| Screen reader / OCR | Phase 3 vision tool covers most of this |
| Reminders beyond calendar | `log_entry` + `add_calendar_event` already cover the use case |

---

## 10. Future / "wouldn't it be cool if" ideas

- Recognize you specifically vs. other people via voice ID.
- Proactive behavior: "you've been at the desk for 4 hours, want a break?"
- Context awareness: knows when you're in a meeting (camera + mic detection) and shuts up.
- Integration with your calendar to give morning briefings.
- A second robot in another room that mirrors state.
- Memory: remembers conversations and facts about you across sessions.
- Mood/affect: changes LED color or voice prosody based on conversational tone.
- Custom skills/plugins: a directory of things it knows how to do, easy to add to.


Ideas:
* When Wakeword is heard, Eyes glow showing visual feedback that the robot is on