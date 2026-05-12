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

MIC_ENABLED_DEFAULT = os.getenv("MIC_ENABLED", "true").lower() == "true"
MAX_UTTERANCE_SECONDS = int(os.getenv("MAX_UTTERANCE_SECONDS", "30"))
CAMERA_LOG_PATH = os.getenv("CAMERA_LOG_PATH", "camera_access.log")
FOLLOWUP_WINDOW_SECONDS = float(os.getenv("FOLLOWUP_WINDOW_SECONDS", "20"))

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


def make_stt_recorder():
    """Build the STT recorder. Returns an object with a .text() method."""
    if STT_PROVIDER == "realtimestt":
        from RealtimeSTT import AudioToTextRecorder
        return AudioToTextRecorder(
            model=STT_MODEL,
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            wake_words="nemo",
            wakeword_backend="oww",
            openwakeword_model_paths="custom_wakewords/nemo.onnx",
            openwakeword_inference_framework="onnx",
        )
    raise ValueError(
        f"Unknown STT_PROVIDER={STT_PROVIDER!r}. "
        f"Set STT_PROVIDER=realtimestt in .env."
    )
