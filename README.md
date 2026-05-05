# Nemo

A voice-controlled desktop assistant. Say the wake word "Nemo," ask a
question or give a command, and it talks back. It can play music on
Spotify, open apps, and answer general questions.

## How it works

- **Wake word + STT** — `RealtimeSTT` listens for "Nemo" and transcribes
  the utterance.
- **Agent** — a LangGraph agent (`agent.py`) runs the conversation and
  decides when to call tools.
- **Tools** — Spotify playback control and a generic open-app tool.
- **TTS** — `RealtimeTTS` streams the response back as audio. Tokens are
  piped into the speech engine as they arrive, so playback starts before
  the model finishes generating.
- **Transport** — `transport.py` is a thin seam between the mic/speaker
  side and the agent side. Today it's an in-process call; the same
  interface lets the two sides run on separate machines later.

## Layout

```
main.py            entry point: mic loop, follow-up window, TTS playback
agent.py           LangGraph agent + system prompt
config.py          provider factories (LLM / TTS / STT)
transport.py       Edge <-> Brain seam
privacy.py         mic gate + camera access log
stt.py, tts.py     thin wrappers over RealtimeSTT / RealtimeTTS
spotify_client.py  Spotify Web API client
tool_manager.py    wires tools into the agent
tools/             Spotify + generic (open-app) tools
engines/           custom TTS engines (Piper)
setup/             one-time bootstrap (Spotify OAuth)
```

## Run it

Uses `uv` and a project-local venv.

```
uv sync
source .venv/bin/activate
python main.py
```

First-time Spotify setup (one-shot, gets you a refresh token):

```
uv run python setup/spotify_oauth_bootstrap.py
# log in at http://127.0.0.1:5000, copy the refresh token into .env
```

## Configuration

Copy `.env.example` to `.env` and fill in the values.

Required for Spotify: `CLIENT_ID`, `CLIENT_SECRET`, `REFRESH_TOKEN`.

Provider selection — providers are explicit, unknown values raise:

```
LLM_PROVIDER=openai      # openai | ollama
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...

TTS_PROVIDER=openai      # openai | coqui | piper
STT_PROVIDER=realtimestt
STT_MODEL=small.en
```

Privacy:

```
MIC_ENABLED=true
MAX_UTTERANCE_SECONDS=30
FOLLOWUP_WINDOW_SECONDS=20
CAMERA_LOG_PATH=camera_access.log
```

See `.env.example` for the full list and notes on Whisper model sizes.

## Roadmap

- **Physical device.** Move the mic/speaker side onto a small dedicated
  board (Pi-class) that talks to the agent over Wi-Fi. The transport
  seam in `transport.py` is already there for this — swap the in-process
  call for a WebSocket and the rest of the code doesn't change. Add a
  camera once the board is stable.
- **Long-term memory.** Persist conversation history and learned facts
  in SQL so Nemo remembers things across restarts (preferences, names,
  recurring tasks, prior context). Today memory lives only in the
  in-process `MemorySaver` for the current session.
- **More tools.** Calendar, reminders, web search, smart-home control.
- **Local-only mode.** Run end-to-end with Ollama + Piper so nothing
  leaves the machine.

