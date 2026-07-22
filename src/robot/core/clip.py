"""ClipService — the "clip that" dashcam loop.

    Screen+SysAudio ─► Swift helper ─► 5s H.264 fMP4 segs ─► RAM disk NemoClipBuf
                       (native/clip-recorder)                     │  256MB
    Mic ─► own sounddevice InputStream ─► 60s PCM deque ──────────┤
                                                                  │
    on_recording_start ──► take_snapshot(): hardlink segs + copy ─┤
    (wake ∙ follow-up ∙ hotkey — single hook)     mic deque       ▼
                                                            save(snapshot):
    watchdog tick (~5s):                                    · +boundary seg
      · newest-seg age <15s else restart + speak once       · timestamp trim
      · TTL-discard snapshot >120s                          · mux + AAC mic
      · TCC permission wall → speak once, NO restart loop         ▼
      · secure-input poll → pause capture               ~/Movies/Nemo Clips/
    MicGate.subscribe ──► gate off: pause + flush everything

Engine facts locked by the Phase 1 gate (prototype/clip-gate/REPORT.md):

- The capture engine is the compiled ScreenCaptureKit helper in
  native/clip-recorder (SCK system-audio tap — no loopback driver), writing
  segmented fMP4 with a FIXED 5s interval and 1s keyframes.
- No on-demand segment flush exists: AVFoundation's flushSegment() requires
  an indefinite interval, which requires passthrough inputs (error -11875).
  save() instead waits ≤~6s for the next natural boundary segment, which
  covers the clip's tail with zero hole (measured 0.91s typical).
- SEG lines carry each segment's earliestPresentationTimeStamp, which is on
  the same mach clock as time.monotonic(). Mic alignment MUST use that PTS,
  never wall-clock arrival times.
- The mic track lags the video clock by ~57ms (MIC_TRIM_S, measured stable
  to ±0.3ms). Re-verify at integration (`just clip-smoke`).
"""

from __future__ import annotations

import collections
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Measured mic-vs-video-clock lag (Phase 1 gate Q6). Positive = the mic track
# runs late; save() trims this much extra off the mic's head to align it.
MIC_TRIM_S = 0.057

MIC_RATE = 16_000

# Spoken failure strings. The save_clip tool and the watchdog speak these
# verbatim — keep them short, first-person, and free of jargon.
SPOKEN_NOT_RUNNING = "Clipping isn't running right now."
SPOKEN_NOTHING_BUFFERED = "I don't have any screen footage buffered yet."
SPOKEN_GATE_WAS_OFF = "My mic gate was off, so there's nothing buffered to clip."
SPOKEN_DEST_UNWRITABLE = "I couldn't write to the clips folder."
SPOKEN_MUX_FAILED = "I couldn't put the clip file together."
SPOKEN_RESTARTED = "Screen capture glitched — I restarted the clip buffer."
SPOKEN_PERMISSION = (
    "I need screen recording permission to keep the clip buffer running."
)
SPOKEN_DISK_FULL = (
    "The clip buffer disk hiccuped — I dropped the oldest footage to keep going."
)
SPOKEN_GAVE_UP = "The clip buffer keeps failing to restart, so I'm leaving it off."


# Keyword fast-path (decision 9A): the canonical phrasings save in <1s with
# no LLM round-trip. Deliberately strict — the WHOLE utterance must be the
# command (plus an optional address/politeness), so mid-sentence mentions
# ("the clip that fell off", "I watched a clip that was funny") and indirect
# requests ("can you clip that for me") fall through to the Brain, where the
# save_clip tool handles paraphrases.
_CLIP_COMMAND_RE = re.compile(
    r"^(?:(?:hey\s+|ok\s+)?nemo[,.!\s]+)?"
    r"(?:please[,\s]+)?"
    r"(?:clip\s+(?:that|this|it)|save\s+(?:that|this|the)\s+clip)"
    r"[,.!?\s]*(?:please)?[.!?\s]*$",
    re.IGNORECASE,
)


def is_clip_command(utterance: str) -> bool:
    """Whether an utterance is a canonical 'clip that' command (fast-path)."""
    return bool(_CLIP_COMMAND_RE.match(utterance.strip()))


class ClipError(RuntimeError):
    """A clip failure whose message is meant to be spoken to the user."""

    def __init__(self, spoken: str):
        super().__init__(spoken)
        self.spoken = spoken


@dataclass(frozen=True)
class Segment:
    seq: int
    path: Path
    size: int
    duration_ms: int
    pts: float  # earliestPresentationTimeStamp — mach clock, = time.monotonic()


@dataclass
class Snapshot:
    """One frozen 60s window: hardlinked segments + a copy of the mic deque.

    Single slot — a newer snapshot replaces this one (the clarify-turn race
    is a documented limitation, eng plan OV #10).
    """

    dir: Path
    init_path: Path
    segments: list[Segment]
    mic_blocks: list
    created_at: float
    generation: int  # which recorder spawn produced these segments


class RamDisk:
    """Named RAM-disk volume with the full orphan-sweep discipline (5A):
    wipe-on-attach if a previous run left it mounted, eject on shutdown.
    RAM (not SSD) because a 24/7 8 Mbps loop writes ~86GB/day — free in RAM,
    real wear on flash — and because unsaved footage should die with power.
    """

    def __init__(self, volume_name: str, size_mb: int, run=subprocess.run):
        self.volume_name = volume_name
        self.size_mb = size_mb
        self._run = run

    @property
    def mountpoint(self) -> Path:
        return Path("/Volumes") / self.volume_name

    def mount(self) -> Path:
        if self.mountpoint.exists():
            log.warning("orphan RAM disk %s found — ejecting", self.mountpoint)
            self._run(
                ["diskutil", "eject", str(self.mountpoint)],
                check=False,
                capture_output=True,
            )
        sectors = self.size_mb * 2048  # 512-byte sectors
        dev = self._run(
            ["hdiutil", "attach", "-nomount", f"ram://{sectors}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self._run(
            ["diskutil", "erasevolume", "HFS+", self.volume_name, dev],
            check=True,
            capture_output=True,
        )
        log.info("clip RAM disk: %sMB at %s (%s)", self.size_mb, self.mountpoint, dev)
        return self.mountpoint

    def unmount(self) -> None:
        if self.mountpoint.exists():
            self._run(
                ["diskutil", "eject", str(self.mountpoint)],
                check=False,
                capture_output=True,
            )


class MicRing:
    """Rolling window of (monotonic_ts, int16 block) from our OWN input
    stream — decision 1A: the ear's recorder is never touched for clip audio.

    Timestamps are taken at callback delivery (≈ block end); alignment code
    accounts for that. If the device disappears (AirPods switch), the
    watchdog's ensure_running() reopens the stream with backoff — clips just
    lose their mic track until it comes back.
    """

    def __init__(
        self,
        rate: int = MIC_RATE,
        window_seconds: float = 60.0,
        stream_factory=None,
        clock=time.monotonic,
    ):
        self.rate = rate
        self.window_seconds = window_seconds
        self._clock = clock
        self._factory = stream_factory or self._default_stream_factory
        self._stream = None
        self._blocks: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._retry_at = 0.0
        self._backoff = 1.0

    def _default_stream_factory(self, callback):
        import sounddevice as sd

        return sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype="int16",
            blocksize=self.rate // 10,
            callback=callback,
        )

    def _on_block(self, indata, frames=None, time_info=None, status=None) -> None:
        now = self._clock()
        with self._lock:
            self._blocks.append((now, np.array(indata, dtype=np.int16, copy=True)))
            horizon = now - self.window_seconds
            while self._blocks and self._blocks[0][0] < horizon:
                self._blocks.popleft()

    def start(self) -> None:
        self.ensure_running()

    def ensure_running(self) -> None:
        """Open (or reopen) the input stream; exponential backoff on failure
        so a missing device doesn't get hammered every watchdog tick."""
        stream = self._stream
        if stream is not None and getattr(stream, "active", True):
            self._backoff = 1.0
            return
        now = self._clock()
        if now < self._retry_at:
            return
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
            self._stream = None
        try:
            new = self._factory(self._on_block)
            new.start()
            self._stream = new
            self._backoff = 1.0
            log.info("clip mic stream open")
        except Exception:
            log.exception("clip mic stream failed to open; retrying later")
            self._retry_at = now + self._backoff
            self._backoff = min(self._backoff * 2, 60.0)

    def snapshot(self) -> list:
        with self._lock:
            return list(self._blocks)

    def flush(self) -> None:
        with self._lock:
            self._blocks.clear()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


def align_mic_samples(blocks: list, start_at: float, rate: int = MIC_RATE):
    """Flatten mic blocks into one array aligned so sample 0 falls at
    monotonic time `start_at` (the clip's first video PTS + MIC_TRIM_S).

    Trim when the mic history reaches back before `start_at`; PAD with
    silence when it starts after (the muxer places the track at t=0, so a
    missing pad shows up as the whole mic track running early — the phantom
    −1.05s offset the gate uncovered).
    """
    audio = np.concatenate([np.asarray(b).reshape(-1) for _, b in blocks])
    first_ts, first_block = blocks[0]
    stream_start = first_ts - len(np.asarray(first_block).reshape(-1)) / rate
    shift = int(round((start_at - stream_start) * rate))
    if shift >= 0:
        return audio[shift:]
    return np.concatenate([np.zeros(-shift, dtype=audio.dtype), audio])


class RecorderProcess:
    """One running instance of the Swift capture helper.

    Owns the stdout pump thread and the 60s segment ring. Segments beyond
    the window are unlinked — hardlinks taken by a snapshot keep the bytes
    alive, so the ring never yanks footage out from under a pending save.
    """

    # The helper's SCK error when Screen Recording permission is missing.
    _PERMISSION_MARKER = "declined tcc"

    def __init__(
        self,
        binary: Path,
        seg_dir: Path,
        *,
        bitrate: int,
        segment_seconds: float,
        window_segments: int,
        popen=subprocess.Popen,
        clock=time.monotonic,
        on_error=None,
    ):
        self.seg_dir = seg_dir
        self.window_segments = window_segments
        self._clock = clock
        self._on_error = on_error
        self.init_path: Path | None = None
        self.segments: collections.deque[Segment] = collections.deque()
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.seg_event = threading.Event()
        self.permission_denied = False
        self.spawned_at = clock()
        self.last_seg_at: float | None = None
        self.proc = popen(
            [
                str(binary),
                "--out",
                str(seg_dir),
                "--bitrate",
                str(bitrate),
                "--segment-seconds",
                str(segment_seconds),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(
            target=self._pump, name="clip-recorder-pump", daemon=True
        ).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "SEG":
                seg = Segment(
                    seq=int(parts[1]),
                    path=Path(parts[2]),
                    size=int(parts[3]),
                    duration_ms=int(parts[4]),
                    pts=float(parts[5]),
                )
                with self.lock:
                    self.segments.append(seg)
                    self.last_seg_at = self._clock()
                    while len(self.segments) > self.window_segments:
                        old = self.segments.popleft()
                        old.path.unlink(missing_ok=True)
                self.seg_event.set()
            elif parts[0] == "INIT":
                self.init_path = Path(parts[1])
            elif parts[0] == "START":
                self.started.set()
            elif parts[0] == "ERR":
                message = line.strip()[4:]
                log.error("clip recorder: %s", message)
                if self._PERMISSION_MARKER in message.lower():
                    self.permission_denied = True
                if self._on_error is not None:
                    try:
                        self._on_error(message)
                    except Exception:
                        log.exception("clip recorder on_error callback raised")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def newest_segment_age(self) -> float:
        """Seconds since the last SEG arrived (since spawn if none yet) —
        the watchdog's health signal for SCK's silent-freeze mode."""
        with self.lock:
            reference = (
                self.last_seg_at if self.last_seg_at is not None else self.spawned_at
            )
        return self._clock() - reference

    def wait_boundary(self, timeout: float) -> bool:
        """Block until the next natural segment boundary lands (≤5s away).
        There is no on-demand flush (see module docstring) — this wait IS
        the save path's tail coverage."""
        self.seg_event.clear()
        return self.seg_event.wait(timeout)

    def snapshot_segments(self) -> list[Segment]:
        with self.lock:
            return list(self.segments)

    def drop_oldest(self, count: int = 1) -> None:
        """Free ring space after a segment-write failure (disk full)."""
        with self.lock:
            for _ in range(count):
                if not self.segments:
                    break
                self.segments.popleft().path.unlink(missing_ok=True)

    def stop(self) -> None:
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=15)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def _secure_input_active() -> bool:
    """Whether secure keyboard entry (password fields) is active system-wide.

    Polled ~per watchdog tick (10A): capture pauses while a password is being
    typed. Fails open to False — a probe failure must not permanently kill
    the clip buffer."""
    try:
        import ctypes

        carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        return bool(carbon.IsSecureEventInputEnabled())
    except Exception:
        return False


class ClipService:
    """Owns the whole loop: RAM disk, capture helper, mic ring, snapshot
    slot, save pipeline, and the health watchdog. Built by
    `config.make_clip_service()`; wired into amain and the MicGate at T5.
    """

    def __init__(
        self,
        *,
        recorder_binary: Path | str,
        save_dir: Path | str,
        volume_name: str = "NemoClipBuf",
        ramdisk_mb: int = 256,
        bitrate: int = 8_000_000,
        segment_seconds: float = 5.0,
        window_seconds: float = 60.0,
        snapshot_ttl: float = 120.0,
        mic_trim_s: float = MIC_TRIM_S,
        stale_after: float = 15.0,
        boundary_timeout: float = 6.0,
        watchdog_interval: float = 5.0,
        max_consecutive_restarts: int = 3,
        speak=None,
        secure_input_check=_secure_input_active,
        clock=time.monotonic,
        run=subprocess.run,
        popen=subprocess.Popen,
        ramdisk: RamDisk | None = None,
        mic_ring: MicRing | None = None,
    ):
        self.recorder_binary = Path(recorder_binary)
        self.save_dir = Path(save_dir).expanduser()
        self.bitrate = bitrate
        self.segment_seconds = segment_seconds
        self.window_segments = max(1, int(round(window_seconds / segment_seconds)))
        self.snapshot_ttl = snapshot_ttl
        self.mic_trim_s = mic_trim_s
        self.stale_after = stale_after
        self.boundary_timeout = boundary_timeout
        self.watchdog_interval = watchdog_interval
        self.max_consecutive_restarts = max_consecutive_restarts
        self._speak_cb = speak
        self._secure_input_check = secure_input_check
        self._clock = clock
        self._run = run
        self._popen = popen
        self._ramdisk = ramdisk or RamDisk(volume_name, ramdisk_mb, run=run)
        self._mic = mic_ring or MicRing(window_seconds=window_seconds, clock=clock)

        self._lock = threading.RLock()
        self._running = False
        self._buf: Path | None = None
        self._recorder: RecorderProcess | None = None
        self._generation = 0
        self._snapshot: Snapshot | None = None
        self._pause_reasons: set[str] = set()
        self._flushed_by_gate = False
        self._spoken: set[str] = set()
        self._consecutive_restarts = 0
        self._gave_up = False
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._buf = self._ramdisk.mount()
            self._mic.start()
            self._spawn_recorder()
            self._running = True
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="clip-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def stop(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=self.watchdog_interval + 2)
            self._watchdog_thread = None
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._recorder is not None:
                self._recorder.stop()
                self._recorder = None
        self._mic.stop()
        self._ramdisk.unmount()

    def attach_gate(self, gate) -> None:
        """Subscribe to the MicGate (3A): gate off = instant pause + flush of
        everything unsaved; gate back on = resume capture."""
        gate.subscribe(self._on_gate)

    def _on_gate(self, enabled: bool) -> None:
        if enabled:
            self.resume("gate")
        else:
            self.pause("gate", flush=True)

    def is_healthy(self) -> bool:
        with self._lock:
            if not self._running or self._gave_up:
                return False
            if self._pause_reasons:
                return True  # deliberately paused, not broken
            rec = self._recorder
        if rec is None or rec.permission_denied or not rec.alive():
            return False
        return rec.newest_segment_age() <= self.stale_after

    # -- capture engine ---------------------------------------------------

    def _spawn_recorder(self) -> None:
        """(Re)start the helper on a clean segment dir. Segment seq numbers
        restart at 1 per spawn, so stale files from the previous run must go;
        snapshot hardlinks live in their own dir and survive the wipe."""
        seg_dir = self._buf / "segments"
        shutil.rmtree(seg_dir, ignore_errors=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        self._generation += 1
        self._recorder = RecorderProcess(
            self.recorder_binary,
            seg_dir,
            bitrate=self.bitrate,
            segment_seconds=self.segment_seconds,
            window_segments=self.window_segments,
            popen=self._popen,
            clock=self._clock,
            on_error=self._on_recorder_error,
        )

    def _on_recorder_error(self, message: str) -> None:
        if "segment write failed" in message.lower():
            rec = self._recorder
            if rec is not None:
                rec.drop_oldest(2)
            self._speak_once("disk", SPOKEN_DISK_FULL)

    def pause(self, reason: str, *, flush: bool = False) -> None:
        """Stop capture. First reason kills the helper (dropping the in-flight
        segment with it); `flush` additionally wipes the ring, mic deque, and
        snapshot — the deafen semantics (gate off = nothing survives)."""
        with self._lock:
            if not self._running:
                return
            first = not self._pause_reasons
            self._pause_reasons.add(reason)
            rec = self._recorder if first else None
            if rec is not None:
                self._recorder = None
        if rec is not None:
            rec.stop()
        if flush:
            self._flush()
            with self._lock:
                self._flushed_by_gate = reason == "gate" or self._flushed_by_gate

    def resume(self, reason: str) -> None:
        # Strictly the inverse of pause(reason): a resume for a reason that
        # never paused us is a no-op — so the watchdog's unconditional
        # secure-input resume can't respawn a recorder that was stood down
        # for a different cause (permission wall, gave-up).
        with self._lock:
            if not self._running or reason not in self._pause_reasons:
                return
            self._pause_reasons.discard(reason)
            if self._pause_reasons or self._recorder is not None:
                return
            self._gave_up = False
            self._consecutive_restarts = 0
            self._spawn_recorder()

    def _flush(self) -> None:
        self._mic.flush()
        with self._lock:
            snap, self._snapshot = self._snapshot, None
            buf = self._buf
        if snap is not None:
            shutil.rmtree(snap.dir, ignore_errors=True)
        if buf is not None:
            seg_dir = buf / "segments"
            shutil.rmtree(seg_dir, ignore_errors=True)
            seg_dir.mkdir(parents=True, exist_ok=True)

    # -- snapshot ---------------------------------------------------------

    def take_snapshot(self) -> None:
        """The `on_recording_start` hook (2A): freeze the current window the
        instant Nemo starts listening — wake word, follow-up bypass, and
        hotkey alike. Runs on the recorder's callback thread, so it must be
        fast (hardlinks + a list copy) and must NEVER raise into the ear."""
        try:
            with self._lock:
                if not self._running or self._pause_reasons:
                    return
                rec = self._recorder
                buf = self._buf
                generation = self._generation
            if rec is None or buf is None or rec.init_path is None:
                return
            segments = rec.snapshot_segments()
            if not segments:
                return
            snap_dir = buf / "snapshot"
            shutil.rmtree(snap_dir, ignore_errors=True)
            snap_dir.mkdir(parents=True)
            init_link = snap_dir / rec.init_path.name
            os.link(rec.init_path, init_link)
            linked = []
            for seg in segments:
                dst = snap_dir / seg.path.name
                os.link(seg.path, dst)
                linked.append(Segment(seg.seq, dst, seg.size, seg.duration_ms, seg.pts))
            snapshot = Snapshot(
                dir=snap_dir,
                init_path=init_link,
                segments=linked,
                mic_blocks=self._mic.snapshot(),
                created_at=self._clock(),
                generation=generation,
            )
            with self._lock:
                self._snapshot = snapshot
                self._flushed_by_gate = False
        except Exception:
            log.exception("clip snapshot failed; continuing without one")

    # -- save -------------------------------------------------------------
    #
    #   wake ─► snapshot ─► user: "clip that" ─► save() called
    #             │                                 │
    #   ring: [s1 … s12] hardlinked                 ├─ wait ≤6s for the next
    #             │                                 │  natural boundary seg
    #   live ring keeps rolling ──► s13 lands ──────┘  (covers the words
    #                                                   spoken while asking)
    #   concat: init + s1…s12 + s13 ─► mux mic (trim/pad to first PTS
    #   + 57ms) ─► ~/Movies/Nemo Clips/clip-<stamp>.mp4

    def save(self) -> Path:
        """Save the pending snapshot as a finished clip. Returns the file
        path; raises ClipError with a speakable message on every failure."""
        with self._lock:
            if not self._running:
                raise ClipError(SPOKEN_NOT_RUNNING)
            snap = self._snapshot
            flushed_by_gate = self._flushed_by_gate
            rec = self._recorder
            generation = self._generation
        if snap is None:
            raise ClipError(
                SPOKEN_GATE_WAS_OFF if flushed_by_gate else SPOKEN_NOTHING_BUFFERED
            )

        # Boundary segment: only meaningful from the same recorder spawn —
        # segments from a restarted helper belong to a different encoder run
        # and can't be concatenated onto this snapshot's init segment.
        boundary: list[Segment] = []
        if rec is not None and generation == snap.generation and rec.alive():
            if not rec.wait_boundary(self.boundary_timeout):
                log.warning("clip save: no boundary segment; tail may be short")
            # Take everything newer than the snapshot regardless — interim
            # segments that landed while the user was talking still belong
            # to this clip even if the fresh boundary never arrived.
            last_snap_seq = snap.segments[-1].seq
            boundary = [s for s in rec.snapshot_segments() if s.seq > last_snap_seq]

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ClipError(SPOKEN_DEST_UNWRITABLE) from exc
        stamp = time.strftime("%Y%m%d-%H%M%S")
        work = self.save_dir / f".work-{stamp}"
        try:
            try:
                work.mkdir()
                raw = work / "raw.mp4"
                with open(raw, "wb") as out:
                    out.write(snap.init_path.read_bytes())
                    for seg in snap.segments:
                        out.write(seg.path.read_bytes())
                    for seg in boundary:
                        out.write(seg.path.read_bytes())
            except OSError as exc:
                raise ClipError(SPOKEN_DEST_UNWRITABLE) from exc

            clip_path = self.save_dir / f"clip-{stamp}.mp4"
            mic_wav = self._write_mic_wav(snap, work)
            if mic_wav is not None:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw),
                    "-i",
                    str(mic_wav),
                    "-map",
                    "0:v",
                    "-map",
                    "0:a",
                    "-map",
                    "1:a",
                    "-c:v",
                    "copy",
                    "-c:a:0",
                    "copy",
                    "-c:a:1",
                    "aac",
                    "-b:a:1",
                    "96k",
                    str(clip_path),
                ]
            else:
                # Mic stream was down for this window (device switch) —
                # degraded clip: video + system audio only.
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw),
                    "-c",
                    "copy",
                    str(clip_path),
                ]
            result = self._run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log.error("clip mux failed: %s", result.stderr)
                raise ClipError(SPOKEN_MUX_FAILED)
        finally:
            shutil.rmtree(work, ignore_errors=True)

        with self._lock:
            if self._snapshot is snap:
                self._snapshot = None
        shutil.rmtree(snap.dir, ignore_errors=True)
        log.info("clip saved: %s", clip_path)
        return clip_path

    def _write_mic_wav(self, snap: Snapshot, work: Path) -> Path | None:
        if not snap.mic_blocks:
            log.warning("clip save: mic ring empty — saving without mic track")
            return None
        # First video PTS shares the mach clock with the mic timestamps;
        # + mic_trim_s drops the measured device latency off the mic's head.
        start_at = snap.segments[0].pts + self.mic_trim_s
        audio = align_mic_samples(snap.mic_blocks, start_at, self._mic.rate)
        mic_wav = work / "mic.wav"
        with wave.open(str(mic_wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self._mic.rate)
            f.writeframes(audio.astype(np.int16).tobytes())
        return mic_wav

    # -- watchdog ---------------------------------------------------------

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self.watchdog_interval):
            try:
                self.tick()
            except Exception:
                log.exception("clip watchdog tick failed")

    def tick(self) -> None:
        """One watchdog pass: snapshot TTL, secure-input gate, capture
        health. Public so tests (and clip-smoke) can drive it directly."""
        with self._lock:
            if not self._running:
                return
            snap = self._snapshot
        if snap is not None and self._clock() - snap.created_at > self.snapshot_ttl:
            with self._lock:
                if self._snapshot is snap:
                    self._snapshot = None
            shutil.rmtree(snap.dir, ignore_errors=True)
            log.info("clip snapshot expired (TTL %ss)", self.snapshot_ttl)

        try:
            secure = bool(self._secure_input_check())
        except Exception:
            secure = False
        if secure:
            self.pause("secure-input")
        else:
            self.resume("secure-input")

        self._mic.ensure_running()

        with self._lock:
            if self._pause_reasons or self._gave_up:
                return
            rec = self._recorder
        if rec is None:
            return

        if rec.permission_denied:
            # A permission wall can't be fixed by respawning — restarting
            # would just re-fail (and re-prompt) forever. Speak, stand down.
            self._speak_once("permission", SPOKEN_PERMISSION)
            with self._lock:
                self._recorder = None
                self._gave_up = True
            rec.stop()
            return

        stale = rec.newest_segment_age() > self.stale_after
        if not rec.alive() or stale:
            self._consecutive_restarts += 1
            if self._consecutive_restarts > self.max_consecutive_restarts:
                self._speak_once("gave-up", SPOKEN_GAVE_UP)
                with self._lock:
                    self._recorder = None
                    self._gave_up = True
                rec.stop()
                return
            log.warning(
                "clip recorder unhealthy (alive=%s stale=%s) — restarting",
                rec.alive(),
                stale,
            )
            rec.stop()
            with self._lock:
                self._spawn_recorder()
            self._speak_once("restart", SPOKEN_RESTARTED)
        else:
            # Healthy: arm the one-shot notices for the next incident.
            self._consecutive_restarts = 0
            self._spoken.discard("restart")
            self._spoken.discard("disk")

    def _speak_once(self, key: str, message: str) -> None:
        """Speak a failure notice once per incident, not once per tick."""
        if key in self._spoken:
            return
        self._spoken.add(key)
        log.warning("clip: %s", message)
        if self._speak_cb is not None:
            try:
                self._speak_cb(message)
            except Exception:
                log.exception("clip speak callback raised")
