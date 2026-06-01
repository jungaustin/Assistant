"""Speech-to-text. Listens for the wake word, returns user utterances."""

from robot.ear.base import Ear
from robot.ear.realtimestt import SpeechToText

__all__ = ["Ear", "SpeechToText"]
