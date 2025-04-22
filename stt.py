import atexit
from RealtimeSTT import AudioToTextRecorder

class SpeechToText:
    def __init__(self):
        self.recorder = AudioToTextRecorder(
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            wakeword_backend="oww",
            openwakeword_model_paths="custom_wakewords/nemo.onnx",
            openwakeword_inference_framework="onnx",
        )
    
    def listen(self):
        return self.recorder.text()