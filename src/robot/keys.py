"""Global hotkeys: physical buttons for the robot on the Keychron.

Page Up and Page Down ARE the robot's buttons. A Quartz event tap
(CGEventTap at the session level) sees every keystroke before apps do,
swallows PgUp/PgDn entirely — no app ever page-scrolls — and dispatches
them as robot controls. When the robot isn't running the keys revert to
normal paging.

  Page Up   — wake toggle. Idle → force-start a listen without the wake
              word (push-to-talk). Already capturing (e.g. a false
              wake-word trigger) → abort and discard, so the robot doesn't
              answer something you never said.
  Page Down — deafen toggle. Flips the MicGate; while off the Edge loop
              won't open the mic at all, wake word included.

Why an event tap and not hidutil UserKeyMapping: on Sequoia (15.6.1) the
mapping attaches to the Keychron K2's HID services but is silently ignored
— likely because the K2 impersonates an Apple keyboard (VendorID 0x5ac) —
so remap-to-F17 never fired. The tap works regardless of what the keyboard
claims to be, and needs no system-level config.

Needs Accessibility permission (System Settings → Privacy & Security) for
whatever process runs the robot; without it CGEventTapCreate returns None
and macOS delivers nothing — hence the loud log at startup either way.

The tap callback must return fast (macOS disables taps that stall the
event stream), so key handling is dispatched to throwaway threads.
Everything the controller touches is already cross-thread safe: MicGate
locks internally, and recorder abort()/start() are the same calls the
watchdog makes from other threads today.
"""

from __future__ import annotations

import threading

from robot.core.logging import get_logger
from robot.voice.beep import cancel_beep, mute_beep, ready_beep, wake_beep

log = get_logger(__name__)

# macOS virtual keycodes, confirmed by capturing the K2's actual events.
VK_WAKE = 116  # Page Up
VK_DEAFEN = 121  # Page Down


class HotkeyController:
    """Key-press semantics, separated from the event tap so tests can drive it."""

    def __init__(self, mic_gate, speech_to_text):
        self.mic_gate = mic_gate
        self.speech_to_text = speech_to_text

    def wake_pressed(self) -> None:
        if not self.mic_gate.enabled:
            # Deafened is an explicit state; the wake key doesn't override
            # it. Low blip = "I heard the key but I'm deafened".
            log.info("hotkey_wake_ignored", reason="mic_deafened")
            cancel_beep()
            return
        if getattr(self.speech_to_text, "is_recording", False):
            # Mid-capture — usually a false wake-word trigger. Discard the
            # audio instead of letting it become an unwanted turn.
            log.info("hotkey_wake_cancel_capture")
            cancel_beep()
            try:
                self.speech_to_text.abort()
            except Exception:
                log.exception("hotkey abort failed")
            return
        log.info("hotkey_wake_force_listen")
        wake_beep()
        try:
            self.speech_to_text.force_start()
        except Exception:
            log.exception("hotkey force_start failed")

    def deafen_pressed(self) -> None:
        enabled = self.mic_gate.toggle()
        log.info("hotkey_deafen_toggle", mic_enabled=enabled)
        if enabled:
            # Same cue as boot: "listening again".
            ready_beep()
            return
        mute_beep()
        # The Edge loop only consults the gate between listens, and it's
        # almost certainly blocked inside listen() right now — abort so the
        # wake-word detector actually stops, instead of deafening only after
        # one more (possibly triggered) capture.
        try:
            self.speech_to_text.abort()
        except Exception:
            log.exception("hotkey abort during deafen failed")


class HotkeyListener:
    """Owns the event-tap thread. Failure to start (e.g. no Accessibility
    permission) is logged, never fatal — the robot must still run voice-only."""

    def __init__(self, controller: HotkeyController):
        self.controller = controller
        self._thread: threading.Thread | None = None
        self._runloop = None
        self._tap = None

    def start(self) -> None:
        try:
            import Quartz
        except Exception:
            log.exception("hotkeys_unavailable_quartz_import_failed")
            return
        self._thread = threading.Thread(
            target=self._run, args=(Quartz,), daemon=True, name="hotkeys"
        )
        self._thread.start()

    def _run(self, Quartz) -> None:
        def callback(proxy, type_, event, refcon):
            # macOS disables a tap it considers stuck; re-enable and move on.
            if type_ in (
                Quartz.kCGEventTapDisabledByTimeout,
                Quartz.kCGEventTapDisabledByUserInput,
            ):
                log.warning("hotkeys_tap_reenabled", reason=type_)
                Quartz.CGEventTapEnable(self._tap, True)
                return event
            vk = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            if vk not in (VK_WAKE, VK_DEAFEN):
                return event
            is_repeat = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventAutorepeat
            )
            # Act on the initial key-down only. Key-ups and autorepeats are
            # swallowed too (returning None below) but must not re-toggle —
            # holding the deafen key would otherwise flip the mic on/off
            # several times a second.
            if type_ == Quartz.kCGEventKeyDown and not is_repeat:
                handler = (
                    self.controller.wake_pressed
                    if vk == VK_WAKE
                    else self.controller.deafen_pressed
                )
                threading.Thread(target=handler, daemon=True).start()
            return None  # swallow: no app ever sees PgUp/PgDn

        mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) | Quartz.CGEventMaskBit(
            Quartz.kCGEventKeyUp
        )
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,  # active tap: may modify/swallow
            mask,
            callback,
            None,
        )
        if tap is None:
            log.warning(
                "hotkeys_unavailable_no_accessibility_permission",
                fix="System Settings → Privacy & Security → Accessibility: "
                "enable the app running the robot, then restart it",
            )
            return
        self._tap = tap
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._runloop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(
            self._runloop, source, Quartz.kCFRunLoopCommonModes
        )
        Quartz.CGEventTapEnable(tap, True)
        log.info("hotkeys_listening", wake="page_up", deafen="page_down")
        Quartz.CFRunLoopRun()

    def stop(self) -> None:
        if self._runloop is not None:
            try:
                import Quartz

                Quartz.CFRunLoopStop(self._runloop)
            except Exception:
                pass
            self._runloop = None
