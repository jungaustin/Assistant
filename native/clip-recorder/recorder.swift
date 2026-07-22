// Nemo clip-that capture helper — the engine ClipService (core/clip.py)
// spawns. Promoted essentially unchanged from the Phase 1 gate prototype
// (prototype/clip-gate/, report in REPORT.md), which locked the engine facts
// this file embodies: fixed 5s fMP4 segments, 1s keyframes, SCK system-audio
// tap, per-segment earliestPresentationTimeStamp on the mach clock.
//
// Captures the display under the mouse + system audio into ~5s fragmented-MP4
// segments (H.264 + AAC) via AVAssetWriter's segmented output, written to
// --out (the NemoClipBuf RAM disk in a real run).
//
// stdout protocol (one line per event, flushed):
//   START <pts-seconds>
//   INIT <path> <bytes>
//   SEG <seq> <path> <bytes> <durationMs> <pts-seconds>
//   ERR <message>
//   DONE
// stdin commands:
//   QUIT   — graceful finish (flushes the final partial segment)
// (No on-demand FLUSH: AVFoundation only allows flushSegment() with
// passthrough inputs — see the comment at preferredOutputSegmentInterval.)
//
// Build + sign:  just build-clip-recorder
// The ad-hoc signature's STABLE identifier (com.nemo.cliprecorder) is what
// lets rebuilds keep their Screen Recording TCC grant without re-prompting.

import AVFoundation
import CoreGraphics
import Foundation
import ScreenCaptureKit
import UniformTypeIdentifiers

struct Args {
    var out = URL(fileURLWithPath: "/tmp/clipproto")
    var bitrate = 8_000_000
    var segmentSeconds = 5.0
    var fps = 30
    var scale = 1.0 // multiplier on native pixel size (0.5 = half resolution)

    init() {
        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let flag = it.next() {
            let value = it.next()
            switch (flag, value) {
            case ("--out", .some(let v)): out = URL(fileURLWithPath: v)
            case ("--bitrate", .some(let v)): bitrate = Int(v) ?? bitrate
            case ("--segment-seconds", .some(let v)): segmentSeconds = Double(v) ?? segmentSeconds
            case ("--fps", .some(let v)): fps = Int(v) ?? fps
            case ("--scale", .some(let v)): scale = Double(v) ?? scale
            default:
                FileHandle.standardError.write(Data("unknown arg \(flag)\n".utf8))
                exit(64)
            }
        }
    }
}

func emit(_ line: String) {
    print(line)
    fflush(stdout)
}

final class SegmentSink: NSObject, AVAssetWriterDelegate {
    let dir: URL
    private var seq = 0

    init(dir: URL) { self.dir = dir }

    func assetWriter(_ writer: AVAssetWriter,
                     didOutputSegmentData segmentData: Data,
                     segmentType: AVAssetSegmentType,
                     segmentReport: AVAssetSegmentReport?) {
        let name: String
        if segmentType == .initialization {
            name = "init.mp4"
        } else {
            seq += 1
            name = String(format: "seg-%05d.m4s", seq)
        }
        let url = dir.appendingPathComponent(name)
        do {
            try segmentData.write(to: url)
        } catch {
            emit("ERR segment write failed: \(error.localizedDescription)")
            return
        }
        if segmentType == .initialization {
            emit("INIT \(url.path) \(segmentData.count)")
        } else {
            let duration = segmentReport?.trackReports.map(\.duration.seconds).max() ?? 0
            // Earliest PTS is on the same mach clock as Python's
            // time.monotonic() — the driver uses it to timestamp-trim the mic.
            let pts = segmentReport?.trackReports
                .map(\.earliestPresentationTimeStamp.seconds).min() ?? -1
            emit("SEG \(seq) \(url.path) \(segmentData.count) \(Int(duration * 1000)) \(pts)")
        }
    }
}

final class CaptureEngine: NSObject, SCStreamOutput, SCStreamDelegate {
    let writer: AVAssetWriter
    let videoIn: AVAssetWriterInput
    let audioIn: AVAssetWriterInput
    private var sessionStarted = false
    private let startLock = NSLock()

    init(writer: AVAssetWriter, videoIn: AVAssetWriterInput, audioIn: AVAssetWriterInput) {
        self.writer = writer
        self.videoIn = videoIn
        self.audioIn = audioIn
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard CMSampleBufferIsValid(sampleBuffer),
              CMSampleBufferDataIsReady(sampleBuffer) else { return }

        switch type {
        case .screen:
            // Only frames marked complete carry displayable pixels; idle/blank
            // status frames must not reach the encoder.
            guard let attachments = CMSampleBufferGetSampleAttachmentsArray(
                    sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
                  let statusRaw = attachments.first?[.status] as? Int,
                  let status = SCFrameStatus(rawValue: statusRaw),
                  status == .complete else { return }

            startLock.lock()
            if !sessionStarted {
                let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                // Required whenever a segment interval is set — even the
                // indefinite/manual mode ("Cannot start file writing" without
                // it). Must match the session start time, so the writer only
                // starts once the first real frame's PTS is known.
                writer.initialSegmentStartTime = pts
                if !writer.startWriting() {
                    startLock.unlock()
                    emit("ERR startWriting failed: \(String(describing: writer.error))")
                    exit(3)
                }
                writer.startSession(atSourceTime: pts)
                sessionStarted = true
                emit("START \(pts.seconds)")
            }
            startLock.unlock()
            if videoIn.isReadyForMoreMediaData, !videoIn.append(sampleBuffer) {
                emit("ERR video append failed: \(writer.error?.localizedDescription ?? "?")")
            }

        case .audio:
            startLock.lock()
            let started = sessionStarted
            startLock.unlock()
            guard started else { return }
            if audioIn.isReadyForMoreMediaData, !audioIn.append(sampleBuffer) {
                emit("ERR audio append failed: \(writer.error?.localizedDescription ?? "?")")
            }

        default:
            break
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        emit("ERR stream stopped: \(error.localizedDescription)")
        exit(4)
    }

    func finish() {
        startLock.lock()
        let started = sessionStarted
        startLock.unlock()
        guard started, writer.status == .writing else { return }
        videoIn.markAsFinished()
        audioIn.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
    }
}

@main
struct Main {
    static func main() async {
        let args = Args()
        do {
            try FileManager.default.createDirectory(at: args.out, withIntermediateDirectories: true)
            try await run(args)
        } catch {
            emit("ERR \(error.localizedDescription)")
            exit(2)
        }
    }

    static func run(_ args: Args) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: true)
        let mouse = CGEvent(source: nil)?.location ?? .zero
        let display = content.displays.first {
            CGDisplayBounds($0.displayID).contains(mouse)
        } ?? content.displays[0]

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let pixelScale = Double(filter.pointPixelScale) * args.scale
        let width = Int(Double(display.width) * pixelScale) / 2 * 2
        let height = Int(Double(display.height) * pixelScale) / 2 * 2

        let cfg = SCStreamConfiguration()
        cfg.width = width
        cfg.height = height
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(args.fps))
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        cfg.showsCursor = true
        cfg.queueDepth = 6
        cfg.capturesAudio = true
        cfg.sampleRate = 48_000
        cfg.channelCount = 2

        let writer = AVAssetWriter(contentType: UTType.mpeg4Movie)
        writer.outputFileTypeProfile = .mpeg4AppleHLS
        // Fixed interval is the ONLY option while the writer encodes:
        // on-demand flushSegment() requires kCMTimeIndefinite, and indefinite
        // mode requires passthrough inputs (AVFoundation -11875 — you'd have
        // to bring your own VideoToolbox encoder). Gate Q3 answer: no
        // on-demand finalize; the save path instead waits ≤5s for the next
        // natural segment boundary, which covers the tail with zero hole.
        writer.preferredOutputSegmentInterval = CMTime(
            seconds: args.segmentSeconds, preferredTimescale: 600)
        let sink = SegmentSink(dir: args.out)
        writer.delegate = sink

        let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: args.bitrate,
                // 1s keyframes: segments can only cut on sync samples, so this
                // bounds both the timer cut jitter and the on-demand FLUSH
                // latency (the <1s tail-hole requirement, gate Q3).
                AVVideoMaxKeyFrameIntervalDurationKey: 1.0,
                AVVideoAllowFrameReorderingKey: false,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ])
        videoIn.expectsMediaDataInRealTime = true

        let audioIn = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 48_000,
            AVNumberOfChannelsKey: 2,
            AVEncoderBitRateKey: 128_000,
        ])
        audioIn.expectsMediaDataInRealTime = true

        guard writer.canAdd(videoIn), writer.canAdd(audioIn) else {
            emit("ERR writer rejected inputs")
            exit(3)
        }
        writer.add(videoIn)
        writer.add(audioIn)

        let engine = CaptureEngine(writer: writer, videoIn: videoIn, audioIn: audioIn)
        let stream = SCStream(filter: filter, configuration: cfg, delegate: engine)
        try stream.addStreamOutput(engine, type: .screen,
                                   sampleHandlerQueue: DispatchQueue(label: "clipproto.video"))
        try stream.addStreamOutput(engine, type: .audio,
                                   sampleHandlerQueue: DispatchQueue(label: "clipproto.audio"))

        emit("DISPLAY \(display.displayID) \(width)x\(height) @\(args.fps)fps \(args.bitrate)bps")
        try await stream.startCapture()

        let quit = DispatchSemaphore(value: 0)

        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let sigint = DispatchSource.makeSignalSource(signal: SIGINT)
        let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM)
        for source in [sigint, sigterm] {
            source.setEventHandler { quit.signal() }
            source.resume()
        }

        DispatchQueue.global().async {
            while let line = readLine(strippingNewline: true) {
                if line == "QUIT" { quit.signal(); return }
            }
            quit.signal() // driver died / stdin closed — shut down cleanly
        }

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                quit.wait()
                cont.resume()
            }
        }

        try? await stream.stopCapture()
        engine.finish() // flushes the final partial segment through the delegate
        emit("DONE")
        exit(0)
    }
}
