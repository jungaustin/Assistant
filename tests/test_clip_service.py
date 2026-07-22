"""Tests for ClipService — the clip-that dashcam loop (eng plan T4).

Everything runs against fakes: a queue-driven fake helper process (no Swift
binary or screen capture), a fake sounddevice stream, a fake monotonic
clock, and a recording subprocess runner (no hdiutil/diskutil/ffmpeg).
Segment files are real files in tmp_path so hardlink semantics are exercised
for real.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot.core.clip import (
    MIC_TRIM_S,
    SPOKEN_DEST_UNWRITABLE,
    SPOKEN_DISK_FULL,
    SPOKEN_GATE_WAS_OFF,
    SPOKEN_GAVE_UP,
    SPOKEN_MUX_FAILED,
    SPOKEN_NOT_RUNNING,
    SPOKEN_NOTHING_BUFFERED,
    SPOKEN_PERMISSION,
    SPOKEN_RESTARTED,
    ClipError,
    ClipService,
    MicRing,
    RamDisk,
    RecorderProcess,
    align_mic_samples,
)

# --- fakes ---------------------------------------------------------------


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeProc:
    """Stands in for the Swift helper Popen: stdout lines come from a queue
    the test feeds; stdin commands are recorded."""

    def __init__(self, cmd=None):
        self.cmd = cmd
        self.lines: queue.Queue = queue.Queue()
        self.commands: list[str] = []
        self.returncode = None
        self.stdin = self

    @property
    def stdout(self):
        return iter(self.lines.get, None)

    def write(self, s: str) -> None:
        self.commands.append(s.strip())

    def flush(self) -> None:
        pass

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        self.lines.put(None)  # unblock the pump thread
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9
        self.lines.put(None)

    def emit(self, line: str) -> None:
        self.lines.put(line)


class FakeStream:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.active = False

    def start(self) -> None:
        if self.fail:
            raise RuntimeError("no input device")
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


class FakeRun:
    """Records every subprocess.run call; fabricates ffmpeg output."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.ffmpeg_rc = 0
        self.raw_bytes = None

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if cmd[0] == "ffmpeg":
            self.raw_bytes = Path(cmd[5]).read_bytes()  # the "-i <raw>" input
            if self.ffmpeg_rc == 0:
                Path(cmd[-1]).write_bytes(b"clip")
            return SimpleNamespace(
                returncode=self.ffmpeg_rc, stdout="", stderr="fake mux error"
            )
        return SimpleNamespace(returncode=0, stdout="/dev/disk9\n", stderr="")


class FakeRamDisk:
    def __init__(self, root: Path):
        self.root = root
        self.mounted = False

    def mount(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.mounted = True
        return self.root

    def unmount(self) -> None:
        self.mounted = False


def wait_until(cond, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return False


# --- environment ---------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    clock = FakeClock()
    procs: list[FakeProc] = []

    def popen(cmd, **kwargs):
        proc = FakeProc(cmd)
        procs.append(proc)
        return proc

    run = FakeRun()
    spoken: list[str] = []
    secure = {"on": False}
    mic = MicRing(
        window_seconds=60,
        stream_factory=lambda cb: FakeStream(),
        clock=clock,
    )
    svc = ClipService(
        recorder_binary=tmp_path / "nemo-clip-recorder",
        save_dir=tmp_path / "clips",
        clock=clock,
        run=run,
        popen=popen,
        ramdisk=FakeRamDisk(tmp_path / "buf"),
        mic_ring=mic,
        speak=spoken.append,
        secure_input_check=lambda: secure["on"],
        watchdog_interval=3600,  # tests drive tick() directly
        boundary_timeout=0.2,
    )
    yield SimpleNamespace(
        svc=svc,
        clock=clock,
        procs=procs,
        run=run,
        spoken=spoken,
        secure=secure,
        mic=mic,
        buf=tmp_path / "buf",
        clips=tmp_path / "clips",
    )
    svc.stop()


def feed_init(env, content: bytes = b"INITSEG"):
    proc = env.procs[-1]
    seg_dir = env.buf / "segments"
    path = seg_dir / "init.mp4"
    path.write_bytes(content)
    proc.emit(f"INIT {path} {len(content)}")
    rec = env.svc._recorder
    assert wait_until(lambda: rec.init_path is not None)
    return path


def feed_segment(env, seq: int, pts: float | None = None, content: bytes | None = None):
    proc = env.procs[-1]
    seg_dir = env.buf / "segments"
    content = content if content is not None else f"SEG{seq}".encode()
    path = seg_dir / f"seg-{seq:05d}.m4s"
    path.write_bytes(content)
    pts = pts if pts is not None else env.clock() - 5.0
    proc.emit(f"SEG {seq} {path} {len(content)} 5000 {pts}")
    rec = env.svc._recorder
    assert wait_until(lambda: any(s.seq == seq for s in rec.snapshot_segments()))
    return path


def feed_mic(env, seconds: float = 1.0):
    block = np.arange(int(env.mic.rate * seconds), dtype=np.int16)
    env.mic._on_block(block)


def started(env):
    env.svc.start()
    feed_init(env)


# --- mic alignment math (gate Q6 lessons) --------------------------------


def test_align_trims_when_mic_starts_before_video():
    # One 1s block ending at t=101 → stream starts at t=100.
    blocks = [(101.0, np.arange(16000, dtype=np.int16))]
    out = align_mic_samples(blocks, start_at=100.25, rate=16000)
    assert len(out) == 12000
    assert out[0] == 4000  # first 0.25s trimmed off


def test_align_pads_with_silence_when_mic_starts_after_video():
    # The phantom −1.05s bug: muxers place the track at t=0, so a late mic
    # start must be padded, not left to drift early.
    blocks = [(101.0, np.arange(16000, dtype=np.int16))]
    out = align_mic_samples(blocks, start_at=99.5, rate=16000)
    assert len(out) == 24000
    assert not out[:8000].any()
    assert out[8000] == 0 and out[8001] == 1  # original audio after the pad


def test_align_spans_multiple_blocks():
    blocks = [
        (100.5, np.zeros(8000, dtype=np.int16)),
        (101.0, np.ones(8000, dtype=np.int16)),
    ]
    out = align_mic_samples(blocks, start_at=100.0, rate=16000)
    assert len(out) == 16000
    assert out[7999] == 0 and out[8000] == 1


# --- MicRing -------------------------------------------------------------


def test_mic_ring_drops_blocks_beyond_window():
    clock = FakeClock()
    ring = MicRing(
        window_seconds=60, stream_factory=lambda cb: FakeStream(), clock=clock
    )
    ring._on_block(np.zeros(1600, dtype=np.int16))
    clock.advance(61)
    ring._on_block(np.ones(1600, dtype=np.int16))
    blocks = ring.snapshot()
    assert len(blocks) == 1
    assert blocks[0][1][0] == 1


def test_mic_ring_flush_empties_and_snapshot_is_a_copy():
    clock = FakeClock()
    ring = MicRing(
        window_seconds=60, stream_factory=lambda cb: FakeStream(), clock=clock
    )
    ring._on_block(np.zeros(1600, dtype=np.int16))
    snap = ring.snapshot()
    ring.flush()
    assert ring.snapshot() == []
    assert len(snap) == 1  # earlier snapshot unaffected


def test_mic_ring_reopens_dead_stream():
    clock = FakeClock()
    streams: list[FakeStream] = []

    def factory(cb):
        streams.append(FakeStream())
        return streams[-1]

    ring = MicRing(stream_factory=factory, clock=clock)
    ring.start()
    assert len(streams) == 1 and streams[0].active
    streams[0].active = False  # device disappeared
    ring.ensure_running()
    assert len(streams) == 2 and streams[1].active


def test_mic_ring_open_failure_backs_off():
    clock = FakeClock()
    attempts = []

    def factory(cb):
        attempts.append(clock())
        return FakeStream(fail=True)

    ring = MicRing(stream_factory=factory, clock=clock)
    ring.start()  # attempt 1 fails, backoff 1s
    ring.ensure_running()  # inside backoff — no attempt
    assert len(attempts) == 1
    clock.advance(1.5)
    ring.ensure_running()  # attempt 2, backoff now 2s
    ring.ensure_running()
    assert len(attempts) == 2
    clock.advance(1.5)
    ring.ensure_running()  # still inside the doubled backoff
    assert len(attempts) == 2
    clock.advance(1.0)
    ring.ensure_running()
    assert len(attempts) == 3


# --- RamDisk -------------------------------------------------------------


def test_ramdisk_mount_attaches_and_erases_with_sector_count():
    run = FakeRun()
    disk = RamDisk("NemoClipBufTestNoSuchVolume", 256, run=run)
    mount = disk.mount()
    assert mount == Path("/Volumes/NemoClipBufTestNoSuchVolume")
    assert run.calls[0][:3] == ["hdiutil", "attach", "-nomount"]
    assert run.calls[0][3] == f"ram://{256 * 2048}"
    assert run.calls[1][:2] == ["diskutil", "erasevolume"]
    assert run.calls[1][4] == "/dev/disk9"


def test_ramdisk_mount_sweeps_orphan_first(tmp_path, monkeypatch):
    run = FakeRun()
    orphan = tmp_path / "vol"
    orphan.mkdir()
    monkeypatch.setattr(RamDisk, "mountpoint", property(lambda self: orphan))
    disk = RamDisk("whatever", 64, run=run)
    disk.mount()
    assert run.calls[0][:2] == ["diskutil", "eject"]


def test_ramdisk_unmount_is_noop_when_not_mounted():
    run = FakeRun()
    disk = RamDisk("NemoClipBufTestNoSuchVolume", 64, run=run)
    disk.unmount()
    assert run.calls == []


# --- RecorderProcess -----------------------------------------------------


def make_recorder(tmp_path, window_segments=3):
    clock = FakeClock()
    proc = FakeProc()
    rec = RecorderProcess(
        tmp_path / "bin",
        tmp_path,
        bitrate=8_000_000,
        segment_seconds=5.0,
        window_segments=window_segments,
        popen=lambda cmd, **kw: proc,
        clock=clock,
    )
    return rec, proc, clock


def test_recorder_ring_trims_and_unlinks_beyond_window(tmp_path):
    rec, proc, _ = make_recorder(tmp_path, window_segments=3)
    paths = []
    for seq in range(1, 5):
        p = tmp_path / f"seg-{seq:05d}.m4s"
        p.write_bytes(b"x")
        paths.append(p)
        proc.emit(f"SEG {seq} {p} 1 5000 {100.0 + seq}")
    assert wait_until(lambda: [s.seq for s in rec.snapshot_segments()] == [2, 3, 4])
    assert not paths[0].exists()  # rolled out of the window and unlinked
    assert all(p.exists() for p in paths[1:])


def test_recorder_parses_pts_and_marks_permission_denial(tmp_path):
    rec, proc, _ = make_recorder(tmp_path)
    p = tmp_path / "seg-00001.m4s"
    p.write_bytes(b"x")
    proc.emit(f"SEG 1 {p} 1 5000 123.456")
    assert wait_until(lambda: rec.snapshot_segments())
    assert rec.snapshot_segments()[0].pts == 123.456
    proc.emit("ERR stream stopped: The user declined TCCs for application")
    assert wait_until(lambda: rec.permission_denied)


def test_recorder_wait_boundary(tmp_path):
    rec, proc, _ = make_recorder(tmp_path)
    assert rec.wait_boundary(0.05) is False

    def emit_later():
        time.sleep(0.05)
        p = tmp_path / "seg-00001.m4s"
        p.write_bytes(b"x")
        proc.emit(f"SEG 1 {p} 1 5000 100.0")

    threading.Thread(target=emit_later).start()
    assert rec.wait_boundary(2.0) is True


def test_recorder_newest_segment_age(tmp_path):
    rec, proc, clock = make_recorder(tmp_path)
    clock.advance(7)
    assert rec.newest_segment_age() == 7  # no segments yet: age since spawn
    p = tmp_path / "seg-00001.m4s"
    p.write_bytes(b"x")
    proc.emit(f"SEG 1 {p} 1 5000 100.0")
    assert wait_until(lambda: rec.snapshot_segments())
    clock.advance(3)
    assert rec.newest_segment_age() == 3


def test_recorder_stop_sends_quit(tmp_path):
    rec, proc, _ = make_recorder(tmp_path)
    rec.stop()
    assert proc.commands == ["QUIT"]
    assert proc.returncode == 0


# --- snapshot ------------------------------------------------------------


def test_snapshot_hardlinks_ring_and_copies_mic(env):
    started(env)
    seg_path = feed_segment(env, 1, pts=990.0)
    feed_mic(env)
    env.svc.take_snapshot()
    snap = env.svc._snapshot
    assert snap is not None
    assert snap.init_path.read_bytes() == b"INITSEG"
    assert [s.seq for s in snap.segments] == [1]
    assert snap.segments[0].path != seg_path  # a hardlink in the slot dir
    assert snap.segments[0].path.read_bytes() == b"SEG1"
    assert len(snap.mic_blocks) == 1


def test_snapshot_survives_ring_rollover(env):
    """The hardlink is the whole point: the live ring may unlink a segment
    long before save() runs, and the snapshot must keep the bytes alive."""
    started(env)
    seg1 = feed_segment(env, 1, pts=990.0)
    env.svc.take_snapshot()
    for seq in range(2, 15):  # roll seg 1 out of the 12-segment window
        feed_segment(env, seq)
    assert not seg1.exists()
    snap = env.svc._snapshot
    assert snap.segments[0].path.read_bytes() == b"SEG1"


def test_snapshot_single_slot_replaced(env):
    started(env)
    feed_segment(env, 1)
    env.svc.take_snapshot()
    feed_segment(env, 2)
    env.svc.take_snapshot()
    snap = env.svc._snapshot
    assert [s.seq for s in snap.segments] == [1, 2]


def test_snapshot_with_empty_ring_is_none(env):
    started(env)
    env.svc.take_snapshot()
    assert env.svc._snapshot is None


def test_snapshot_never_raises_into_the_ear(env):
    started(env)
    feed_segment(env, 1)
    env.svc._recorder.init_path = env.buf / "segments" / "gone.mp4"
    env.svc.take_snapshot()  # os.link fails; must swallow
    assert env.svc._snapshot is None


# --- save ----------------------------------------------------------------


def save_in_background(env):
    """save() blocks in wait_boundary; run it on a thread so the test can
    feed the boundary segment mid-wait."""
    result: dict = {}

    def run():
        try:
            result["path"] = env.svc.save()
        except ClipError as exc:
            result["error"] = exc.spoken

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def test_save_happy_path_includes_boundary_segment(env):
    started(env)
    feed_segment(env, 1, pts=990.0)
    feed_segment(env, 2, pts=995.0)
    feed_mic(env)
    env.svc.take_snapshot()
    thread, result = save_in_background(env)
    time.sleep(0.05)
    feed_segment(env, 3, pts=1000.0)  # the natural boundary
    thread.join(timeout=5)
    assert "path" in result, result
    assert result["path"].exists()
    assert env.run.raw_bytes == b"INITSEG" + b"SEG1" + b"SEG2" + b"SEG3"
    cmd = env.run.calls[-1]
    assert cmd[0] == "ffmpeg" and cmd.count("-i") == 2  # raw + mic wav
    assert env.svc._snapshot is None  # slot consumed


def test_save_proceeds_when_boundary_never_arrives(env):
    started(env)
    feed_segment(env, 1, pts=990.0)
    feed_mic(env)
    env.svc.take_snapshot()
    clip = env.svc.save()  # waits boundary_timeout=0.2s, then proceeds
    assert clip.exists()
    assert env.run.raw_bytes == b"INITSEG" + b"SEG1"


def test_save_without_mic_is_degraded_not_fatal(env):
    started(env)
    feed_segment(env, 1)
    env.svc.take_snapshot()  # mic ring empty (device was down)
    clip = env.svc.save()
    assert clip.exists()
    cmd = env.run.calls[-1]
    assert cmd.count("-i") == 1  # no mic track muxed


def test_save_mic_wav_aligned_to_first_pts_plus_trim(env, monkeypatch):
    seen = {}

    def spy(blocks, start_at, rate):
        seen["start_at"] = start_at
        return np.zeros(16, dtype=np.int16)

    monkeypatch.setattr("robot.core.clip.align_mic_samples", spy)
    started(env)
    feed_segment(env, 1, pts=990.0)
    feed_mic(env)
    env.svc.take_snapshot()
    env.svc.save()
    assert seen["start_at"] == pytest.approx(990.0 + MIC_TRIM_S)


def test_save_with_no_snapshot_speaks_nothing_buffered(env):
    started(env)
    with pytest.raises(ClipError) as exc:
        env.svc.save()
    assert exc.value.spoken == SPOKEN_NOTHING_BUFFERED


def test_save_when_not_running_speaks_not_running(env):
    with pytest.raises(ClipError) as exc:
        env.svc.save()
    assert exc.value.spoken == SPOKEN_NOT_RUNNING


def test_save_after_gate_flush_speaks_gate_phrase(env):
    started(env)
    feed_segment(env, 1)
    env.svc.take_snapshot()
    env.svc._on_gate(False)
    env.svc._on_gate(True)
    with pytest.raises(ClipError) as exc:
        env.svc.save()
    assert exc.value.spoken == SPOKEN_GATE_WAS_OFF


def test_save_when_mux_fails_speaks_and_keeps_snapshot(env):
    started(env)
    feed_segment(env, 1)
    env.svc.take_snapshot()
    env.run.ffmpeg_rc = 1
    with pytest.raises(ClipError) as exc:
        env.svc.save()
    assert exc.value.spoken == SPOKEN_MUX_FAILED
    assert env.svc._snapshot is not None  # not consumed — retry possible


def test_save_when_dest_unwritable_speaks(env):
    started(env)
    feed_segment(env, 1)
    env.svc.take_snapshot()
    env.clips.write_bytes(b"a file where the folder should be")
    with pytest.raises(ClipError) as exc:
        env.svc.save()
    assert exc.value.spoken == SPOKEN_DEST_UNWRITABLE


def test_save_skips_boundary_from_a_restarted_recorder(env):
    """Segments from a respawned helper belong to a different encoder run —
    concatenating them onto the old init segment would corrupt the clip."""
    started(env)
    feed_segment(env, 1, pts=990.0)
    env.svc.take_snapshot()
    env.svc.pause("secure-input")  # no flush — snapshot survives
    env.svc.resume("secure-input")
    feed_init(env, b"INIT2")
    feed_segment(env, 1, pts=1010.0, content=b"NEWRUN")
    clip = env.svc.save()
    assert clip.exists()
    assert env.run.raw_bytes == b"INITSEG" + b"SEG1"  # old run only


# --- gate / pause / flush ------------------------------------------------


def test_gate_off_pauses_and_flushes_everything(env):
    started(env)
    feed_segment(env, 1)
    feed_mic(env)
    env.svc.take_snapshot()
    env.svc._on_gate(False)
    assert "QUIT" in env.procs[0].commands  # helper stopped
    assert env.svc._snapshot is None
    assert env.mic.snapshot() == []
    assert list((env.buf / "segments").iterdir()) == []


def test_gate_on_resumes_with_a_fresh_recorder(env):
    started(env)
    env.svc._on_gate(False)
    assert len(env.procs) == 1
    env.svc._on_gate(True)
    assert len(env.procs) == 2


def test_attach_gate_wires_real_micgate(env):
    from robot.privacy.gate import MicGate

    started(env)
    gate = MicGate(enabled=True)
    env.svc.attach_gate(gate)
    gate.set(False)
    assert "QUIT" in env.procs[0].commands
    gate.set(True)
    assert len(env.procs) == 2


def test_secure_input_pauses_without_flushing_history(env):
    started(env)
    feed_segment(env, 1)
    feed_mic(env)
    env.svc.take_snapshot()
    env.secure["on"] = True
    env.svc.tick()
    assert "QUIT" in env.procs[0].commands
    assert env.svc._snapshot is not None  # secure pause keeps the snapshot
    assert env.mic.snapshot() != []
    env.secure["on"] = False
    env.svc.tick()
    assert len(env.procs) == 2  # capture resumed


def test_resume_for_a_reason_that_never_paused_is_a_noop(env):
    started(env)
    env.svc.resume("secure-input")
    assert len(env.procs) == 1


# --- watchdog ------------------------------------------------------------


def test_tick_discards_expired_snapshot(env):
    started(env)
    feed_segment(env, 1)
    env.svc.take_snapshot()
    snap_dir = env.svc._snapshot.dir
    env.clock.advance(121)
    env.svc.tick()
    assert env.svc._snapshot is None
    assert not snap_dir.exists()


def test_tick_keeps_fresh_snapshot(env):
    started(env)
    feed_segment(env, 1)
    env.svc.take_snapshot()
    env.svc.tick()
    assert env.svc._snapshot is not None


def test_stale_capture_restarts_and_speaks_once(env):
    started(env)
    feed_segment(env, 1)
    env.clock.advance(20)  # > stale_after
    env.svc.tick()
    assert len(env.procs) == 2  # restarted
    assert env.spoken == [SPOKEN_RESTARTED]
    env.clock.advance(20)  # new helper produced nothing either
    env.svc.tick()
    assert len(env.procs) == 3
    assert env.spoken == [SPOKEN_RESTARTED]  # once per incident, not per tick


def test_recovery_rearms_the_restart_notice(env):
    started(env)
    feed_segment(env, 1)
    env.clock.advance(20)
    env.svc.tick()  # incident 1 → restart + speak
    feed_init(env)
    feed_segment(env, 1)
    env.svc.tick()  # healthy tick re-arms the notice
    env.clock.advance(20)
    env.svc.tick()  # incident 2
    assert env.spoken == [SPOKEN_RESTARTED, SPOKEN_RESTARTED]


def test_restart_loop_gives_up_after_cap(env):
    started(env)
    for _ in range(10):
        env.clock.advance(20)
        env.svc.tick()
    # initial spawn + max_consecutive_restarts respawns, then stand-down.
    assert len(env.procs) == 1 + env.svc.max_consecutive_restarts
    assert env.spoken == [SPOKEN_RESTARTED, SPOKEN_GAVE_UP]
    assert env.svc.is_healthy() is False


def test_permission_wall_speaks_once_and_never_restart_loops(env):
    started(env)
    env.procs[0].emit("ERR stream stopped: The user declined TCCs for application")
    assert wait_until(lambda: env.svc._recorder.permission_denied)
    for _ in range(3):
        env.svc.tick()
    assert len(env.procs) == 1  # no respawn against the permission wall
    assert env.spoken == [SPOKEN_PERMISSION]
    assert env.svc.is_healthy() is False


def test_segment_write_failure_drops_oldest_and_speaks_once(env):
    started(env)
    for seq in range(1, 4):
        feed_segment(env, seq)
    rec = env.svc._recorder
    env.procs[0].emit("ERR segment write failed: no space left")
    assert wait_until(lambda: len(rec.snapshot_segments()) == 1)
    env.procs[0].emit("ERR segment write failed: no space left")
    assert wait_until(lambda: SPOKEN_DISK_FULL in env.spoken)
    assert env.spoken == [SPOKEN_DISK_FULL]


def test_tick_reopens_dead_mic_stream(env):
    started(env)
    feed_segment(env, 1)
    env.mic._stream.active = False
    env.svc.tick()
    assert env.mic._stream.active


# --- lifecycle / health --------------------------------------------------


def test_start_mounts_spawns_and_stop_tears_down(env):
    env.svc.start()
    assert env.svc._ramdisk.mounted
    assert len(env.procs) == 1
    assert env.procs[0].cmd[0].endswith("nemo-clip-recorder")
    env.svc.start()  # idempotent
    assert len(env.procs) == 1
    env.svc.stop()
    assert not env.svc._ramdisk.mounted
    assert "QUIT" in env.procs[0].commands
    env.svc.stop()  # idempotent


def test_is_healthy_tracks_segment_freshness(env):
    started(env)
    feed_segment(env, 1)
    assert env.svc.is_healthy() is True
    env.clock.advance(20)
    assert env.svc.is_healthy() is False


def test_is_healthy_true_while_deliberately_paused(env):
    started(env)
    feed_segment(env, 1)
    env.svc.pause("secure-input")
    env.clock.advance(999)
    assert env.svc.is_healthy() is True  # paused is not broken


def test_is_healthy_false_before_start(env):
    assert env.svc.is_healthy() is False
