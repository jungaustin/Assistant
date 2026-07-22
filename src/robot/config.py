import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai-compat").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OpenAI-compatible base URL. Defaults to OpenAI itself; point at Ollama
# (http://localhost:11434/v1), Together, vLLM, etc. to swap providers.
BRAIN_BASE_URL = os.getenv("BRAIN_BASE_URL") or None
BRAIN_API_KEY = os.getenv("BRAIN_API_KEY") or os.getenv("OPENAI_API_KEY")

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "")
PIPER_VOICE_PATH = os.getenv(
    "PIPER_VOICE_PATH",
    "models/tts/en-us-amy-medium/en_US-amy-medium.onnx",
)

STT_PROVIDER = os.getenv("STT_PROVIDER", "realtimestt").lower()
STT_MODEL = os.getenv("STT_MODEL", "small.en")
# End-of-utterance tuning. post_speech_silence_duration is how long the
# recorder waits in silence before deciding you've finished talking;
# RealtimeSTT's 0.6s default cuts people off on a normal mid-sentence pause,
# so 1.3s leaves room to breathe. webrtc_sensitivity is VAD aggressiveness
# 0–3 (higher = more eager to call speech "silence"); 2 is gentler than the
# library default of 3, which helps with quiet or trailing-off speech.
STT_END_SILENCE_SECONDS = float(os.getenv("STT_END_SILENCE_SECONDS", "1.3"))
STT_WEBRTC_SENSITIVITY = int(os.getenv("STT_WEBRTC_SENSITIVITY", "2"))
# Use Silero (a neural VAD) instead of WebRTC alone to decide when speech has
# *ended*. WebRTC at a gentle sensitivity gets fooled by breath/room tone and
# sometimes never registers the silence, so the recorder keeps capturing after
# you've finished. Silero is far better at speech-vs-not, which is what end-of-
# turn detection actually needs. The Silero model is already loaded today (it's
# used for start-of-speech), so this only changes which detector ends the turn.
STT_SILERO_DEACTIVITY = os.getenv("STT_SILERO_DEACTIVITY", "true").lower() == "true"
# Load the ONNX build of Silero rather than the torch one. Off by default to
# match the existing load path exactly; flip on for a lighter CPU footprint.
STT_SILERO_USE_ONNX = os.getenv("STT_SILERO_USE_ONNX", "false").lower() == "true"

# Pin the capture device by (case-insensitive substring of) its PyAudio name,
# e.g. "MacBook Pro Microphone". Empty = system default input. Pinning matters
# with Bluetooth headphones: if the headset becomes the default input and the
# robot captures from it, macOS drops the headset into low-quality HFP mode.
# Pinning the built-in mic keeps the headset output-only (crisp A2DP audio).
# Matched by name, not index — indices shift as devices connect/disconnect.
STT_INPUT_DEVICE = os.getenv("STT_INPUT_DEVICE", "")

MIC_ENABLED_DEFAULT = os.getenv("MIC_ENABLED", "true").lower() == "true"
MAX_UTTERANCE_SECONDS = int(os.getenv("MAX_UTTERANCE_SECONDS", "30"))
CAMERA_LOG_PATH = os.getenv("CAMERA_LOG_PATH", "camera_access.log")
FOLLOWUP_WINDOW_SECONDS = float(os.getenv("FOLLOWUP_WINDOW_SECONDS", "8"))
# Quiet gap inserted after Nemo finishes speaking before the follow-up window
# opens. Lets the speaker's audio tail / room echo die down so the mic doesn't
# capture Nemo's own voice and feed it back as a "user" turn (the self-talk
# loop). Small on purpose — it's dead air the user waits through.
FOLLOWUP_POST_SPEECH_DELAY = float(os.getenv("FOLLOWUP_POST_SPEECH_DELAY", "0.4"))
# How similar a follow-up transcript must be to what Nemo just said before we
# treat it as an echo of our own speech and discard it instead of answering.
# Only applied to utterances that already span most of the spoken reply (see
# _is_echo_or_junk), so it targets near-whole echoes, not short new commands.
# Set above 1.0 to disable fuzzy echo matching entirely (verbatim/substring
# echoes are still caught).
ECHO_SIMILARITY_THRESHOLD = float(os.getenv("ECHO_SIMILARITY_THRESHOLD", "0.9"))
# How often the listen() watchdog polls recorder health. RealtimeSTT's capture
# loop runs in a daemon thread; if it dies, listen() blocks forever. The
# watchdog notices the dead thread and rebuilds the recorder. Also the fast-path
# poll interval, so keep it small (a completed listen returns immediately, not
# after this delay).
STT_HEALTH_POLL_SECONDS = float(os.getenv("STT_HEALTH_POLL_SECONDS", "0.5"))
# How often, while a wake-word listen is still blocked, to emit a heartbeat log
# line. Turns an indefinite wait into something visible: regular heartbeats mean
# "alive, waiting for the wake word"; heartbeats that stop (or flip to
# healthy=False) mean the recorder wedged. Set high enough not to spam an idle
# robot's logs.
STT_LISTEN_HEARTBEAT_SECONDS = float(os.getenv("STT_LISTEN_HEARTBEAT_SECONDS", "30"))

# Transport seam (Phase 8 Pi/Mac split). "inproc" runs Brain and Edge in one
# process — today's default. "websocket" makes `python -m robot.main` run
# Edge-only and connect to a brain server (`python -m robot.brain_server`).
TRANSPORT = os.getenv("TRANSPORT", "inproc").lower()
# Client side: where the Edge finds the brain server.
BRAIN_WS_URL = os.getenv("BRAIN_WS_URL", "ws://localhost:8765")
# Server side: bind address. localhost by default (loopback smoke test);
# set 0.0.0.0 in .env when the Pi needs to reach it over the LAN.
BRAIN_WS_HOST = os.getenv("BRAIN_WS_HOST", "localhost")
BRAIN_WS_PORT = int(os.getenv("BRAIN_WS_PORT", "8765"))
# Shared secret both sides must present/verify. Required whenever
# TRANSPORT=websocket — the server refuses to start without it, so an open
# LAN port can never expose an unauthenticated brain.
TRANSPORT_TOKEN = os.getenv("TRANSPORT_TOKEN", "")

# How many recent messages to send to the LLM per call. The full conversation
# is still checkpointed and searchable via recall(); this only bounds the
# per-call prompt so latency doesn't grow as the day's thread accumulates
# (a tool turn = two LLM round-trips over the whole history). Keep comfortably
# above one turn's worth of tool messages so a single turn is never cut.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))

# Conversation state. SqliteSaver writes here so the process survives
# restart with its conversation memory intact. Directory is created on
# first use; the file is git-ignored (state/ in .gitignore).
STATE_DB_PATH = os.getenv("STATE_DB_PATH", "state/conversations.db")

# Durable episodic memory (the MVM episodic log — see memory-architecture.md).
# Separate file from STATE_DB_PATH on purpose: the checkpointer's connection is
# driven by LangGraph's threadpool, and episodic writes shouldn't contend with
# it. `recall(query)` searches this store.
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "state/memory.db")

# Personal data tracker (calories, exercise, sleep, mood, period notes).
TRACKER_DB_PATH = os.getenv("TRACKER_DB_PATH", "state/tracker.db")

# Nightly check-in (proactive, see core/checkin.py). At this local hour,
# Nemo asks about any required types not yet logged today. Disable with
# hour=-1 or an empty types list.
DAILY_LOG_PROMPT_HOUR = int(os.getenv("DAILY_LOG_PROMPT_HOUR", "22"))
DAILY_REQUIRED_TYPES = [
    t.strip().lower()
    for t in os.getenv("DAILY_REQUIRED_TYPES", "calories,exercise,sleep").split(",")
    if t.strip()
]

# Clip-that (the "clip that" dashcam loop — core/clip.py). Off by default:
# it opens a second mic stream and a screen capture at boot, which should be
# a deliberate opt-in per machine (and needs Screen Recording TCC granted to
# whatever launches the robot).
CLIP_ENABLED = os.getenv("CLIP_ENABLED", "false").lower() == "true"
# Engine numbers below are the Phase 1 gate results (prototype/clip-gate/
# REPORT.md) — the 5s segment interval especially is a locked engine fact
# (fixed-interval fMP4; save rides the boundary), not a tuning knob.
CLIP_SEGMENT_SECONDS = float(os.getenv("CLIP_SEGMENT_SECONDS", "5"))
CLIP_WINDOW_SECONDS = float(os.getenv("CLIP_WINDOW_SECONDS", "60"))
# 8 Mbps kept terminal/browser text legible on the QHD display at the gate;
# if the built-in Retina panel ever looks soft, raise this first (the RAM
# disk budget below may grow to 512 with it).
CLIP_BITRATE = int(os.getenv("CLIP_BITRATE", "8000000"))
CLIP_RAMDISK_MB = int(os.getenv("CLIP_RAMDISK_MB", "256"))
CLIP_SNAPSHOT_TTL = float(os.getenv("CLIP_SNAPSHOT_TTL", "120"))
CLIP_SAVE_DIR = os.getenv("CLIP_SAVE_DIR", "~/Movies/Nemo Clips")
# Relative to robot/ (the justfile cwd), like PERSONA_PATH and the model
# paths. Built by `just build-clip-recorder`.
CLIP_RECORDER_BIN = os.getenv(
    "CLIP_RECORDER_BIN", "native/clip-recorder/nemo-clip-recorder"
)


def make_clip_service(speak=None):
    """Build the ClipService from CLIP_* config (clip plan decision 4A).

    `speak` is the fire-and-forget voice callback the watchdog uses for its
    once-per-incident failure notices. The caller owns lifecycle: start()
    after boot, attach_gate() on the MicGate, stop() on shutdown.
    """
    from robot.core.clip import ClipService

    return ClipService(
        recorder_binary=CLIP_RECORDER_BIN,
        save_dir=CLIP_SAVE_DIR,
        ramdisk_mb=CLIP_RAMDISK_MB,
        bitrate=CLIP_BITRATE,
        segment_seconds=CLIP_SEGMENT_SECONDS,
        window_seconds=CLIP_WINDOW_SECONDS,
        snapshot_ttl=CLIP_SNAPSHOT_TTL,
        speak=speak,
    )


# Tavily web search API key. Get one at https://tavily.com.
# Used by the web_search tool for voice-optimized answer synthesis.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Discord bot token (send + catch-up tools). Created at
# https://discord.com/developers/applications — see .env.example for the
# one-time setup steps. Tools degrade gracefully when unset.
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
# Per-channel "last message Nemo summarized" markers. Discord doesn't share
# your personal read state with bots, so Nemo tracks its own.
DISCORD_CURSOR_PATH = os.getenv("DISCORD_CURSOR_PATH", "state/discord_cursors.json")

# Google Calendar. credentials.json is downloaded once from the Google
# Cloud Console (Desktop app OAuth client). The token file is written by
# the OAuth bootstrap and refreshed automatically on use. Both live in
# state/ so they're gitignored.
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
GOOGLE_CALENDAR_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_CALENDAR_CREDENTIALS_PATH", "state/google-credentials.json"
)
GOOGLE_CALENDAR_TOKEN_PATH = os.getenv(
    "GOOGLE_CALENDAR_TOKEN_PATH", "state/google-calendar-token.json"
)


def daily_thread_id() -> str:
    """Today's local date as an ISO string — the default `thread_id` policy.

    Local time, not UTC: a desk robot's "today" should match wall-clock.
    Otherwise UTC midnight could wipe your morning conversation mid-afternoon
    depending on your timezone.

    Why per-day: matches how humans remember small daily interactions.
    Within a day, follow-up turns and mid-afternoon callbacks work
    naturally. Across days, conversations reset — cleaner mental model
    than "this thread spans a week because you forgot to start a new one".
    Cross-day search is the job of recall(), landing in Phase 6b.

    Tests and one-off scripts can override with `Agent(thread_id=...)`.
    """
    from datetime import date

    return date.today().isoformat()


# Persona system prompt. Swap the file (or PERSONA_PATH) to give the agent a
# different personality without touching code.
PERSONA_PATH = os.getenv("PERSONA_PATH", "personas/nemo.md")


def load_persona() -> str:
    """Read the persona file and return its contents.

    Resolved relative to the current working directory (justfile runs from
    robot/, where personas/ lives). Raises FileNotFoundError with a clear
    message if the file is missing — silent fallback to a hardcoded prompt
    would mask the misconfiguration.
    """
    path = Path(PERSONA_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Persona file not found at {path.resolve()}. "
            f"Set PERSONA_PATH in .env or create the file."
        )
    return path.read_text(encoding="utf-8")


def make_llm():
    """Build the chat LLM. No silent default to cloud — provider is explicit."""
    if LLM_PROVIDER == "openai-compat":
        from robot.brain.openai_compat import OpenAICompatChat

        return OpenAICompatChat(
            model=LLM_MODEL,
            base_url=BRAIN_BASE_URL,
            api_key=BRAIN_API_KEY,
        )
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    raise ValueError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. "
        f"Set LLM_PROVIDER=openai-compat or LLM_PROVIDER=ollama in .env."
    )


def make_tts_engine():
    """Build the TTS engine. Returns a RealtimeTTS engine instance."""
    if TTS_PROVIDER == "openai":
        from RealtimeTTS import OpenAIEngine

        return OpenAIEngine(voice=TTS_VOICE) if TTS_VOICE else OpenAIEngine()
    if TTS_PROVIDER == "coqui":
        from RealtimeTTS import CoquiEngine

        return CoquiEngine()
    if TTS_PROVIDER == "piper":
        from robot.voice.engines.piper_engine import PiperEngine

        return PiperEngine(voice_path=PIPER_VOICE_PATH)
    raise ValueError(
        f"Unknown TTS_PROVIDER={TTS_PROVIDER!r}. "
        f"Set TTS_PROVIDER=openai, coqui, or piper in .env."
    )


def resolve_input_device_index(name_fragment: str) -> int | None:
    """Find a PyAudio input device whose name contains `name_fragment`.

    Case-insensitive substring match over input-capable devices only (a
    Bluetooth headset shows up as both input and output; matching on outputs
    could pin playback hardware as a mic). Returns the device index, or None
    when the fragment is empty or nothing matches — the caller falls back to
    the system default input, because a missing device (headset off, mic
    renamed) must degrade to "works like before", never refuse to listen.

    Resolved fresh on every recorder build (start-up and watchdog restarts),
    so the index is correct even though CoreAudio renumbers devices as they
    connect and disconnect.
    """
    if not name_fragment:
        return None
    import pyaudio

    needle = name_fragment.strip().lower()
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if (
                info.get("maxInputChannels", 0) > 0
                and needle in str(info.get("name", "")).lower()
            ):
                return i
    finally:
        p.terminate()
    return None


def make_stt_recorder(on_recording_start=None):
    """Build the STT recorder. Returns an object with a .text() method.

    `on_recording_start` fires when capture actually begins — wake word,
    follow-up-window bypass, or hotkey force_start() alike. That single hook
    is the clip service's snapshot trigger (clip plan decision 2A): any moment
    Nemo starts listening is a moment "clip that" might be said about the
    preceding minute.
    """
    if STT_PROVIDER == "realtimestt":
        import logging

        from RealtimeSTT import AudioToTextRecorder

        from robot.voice.beep import wake_beep

        # Pin the mic (see STT_INPUT_DEVICE). None = system default input.
        input_device_index = resolve_input_device_index(STT_INPUT_DEVICE)
        if STT_INPUT_DEVICE and input_device_index is None:
            logging.getLogger(__name__).warning(
                "stt input device %r not found; using system default input",
                STT_INPUT_DEVICE,
            )

        return AudioToTextRecorder(
            model=STT_MODEL,
            input_device_index=input_device_index,
            # No terminal spinner: it's redundant with the beeps/logs, and
            # RealtimeSTT's abort() sets state to "transcribing" even when
            # nothing was captured, so the spinner printed a misleading
            # "transcribing" every time the deafen key aborted a listen.
            spinner=False,
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            post_speech_silence_duration=STT_END_SILENCE_SECONDS,
            webrtc_sensitivity=STT_WEBRTC_SENSITIVITY,
            silero_deactivity_detection=STT_SILERO_DEACTIVITY,
            silero_use_onnx=STT_SILERO_USE_ONNX,
            wake_words="nemo",
            # Same cue as the push-to-talk key: "I heard you, talk now."
            # Beeps are fire-and-forget on their own thread/stream, so this
            # can't stall the capture thread or clip the start of speech.
            # Doesn't fire on force_start() (the hotkey beeps itself) or
            # during a follow-up window's wake-word bypass.
            on_wakeword_detected=wake_beep,
            wakeword_backend="oww",
            openwakeword_model_paths="models/wake/nemo.onnx",
            openwakeword_inference_framework="onnx",
            on_recording_start=on_recording_start,
        )
    raise ValueError(
        f"Unknown STT_PROVIDER={STT_PROVIDER!r}. "
        f"Set STT_PROVIDER=realtimestt in .env."
    )
