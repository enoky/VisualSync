#!/usr/bin/env python3
"""
Beat-synced dance video builder.

Takes the (equal-length) dancer clips in ./video, retimes every clip to a whole
number of bars at the DJ Mix's BPM, loops the sequence for the full length of
the mix in ./audio, and renders a single 1080p MP4 with the mix as its audio
track, finishing with a fade to black.

Everything happens in ONE ffmpeg pass, with no intermediate files:

    concat demuxer (clip list, repeated)  ->  NVDEC decode (GPU)
      -> setpts  : frame n is placed at n * (segment / frames_per_clip),
                   so every clip spans exactly N bars and there is no drift
      -> fps     : resample onto a constant 24 fps grid (drops frames when
                   speeding up, duplicates when slowing down)
      -> scale_cuda (GPU, lanczos) -> download -> centre crop to 1920x1080
      -> fade to black -> NVENC encode (GPU)
    DJ Mix -> [optional tempo change] -> AAC 320 kbps -> muxed into MP4;
    output cut at the (re-tempo'd) mix's length

The render is written to "<name>.partial.mp4" and renamed when it finishes,
so a cancelled or failed render never leaves a broken file behind.

Usage (or use the GUI: run_gui.bat / gui.py):
    .venv\\Scripts\\python make_dance_video.py --bpm 125
    .venv\\Scripts\\python make_dance_video.py            (asks for the BPMs)
    .venv\\Scripts\\python make_dance_video.py --bpm 124 --preview 60
    .venv\\Scripts\\python make_dance_video.py --bpm 124 --target-bpm 126
    .venv\\Scripts\\python make_dance_video.py --bpm 124 --target-bpm 126 --stretch varispeed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # the script still works without the progress bar
    tqdm = None

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff", ".mp3", ".m4a", ".ogg", ".opus"}
SOXR = "aresample=48000:resampler=soxr:precision=28"

# Stop every ffprobe call flashing a console window when run from the GUI (pythonw).
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class RenderError(Exception):
    """A problem with the inputs or settings, with a message meant for the user."""


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClipInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    frames: int

    @property
    def duration(self) -> float:
        return float(self.frames / self.fps)


@dataclass(frozen=True)
class Inputs:
    """Everything learned from the source files (the slow part, done once)."""
    clips: tuple[ClipInfo, ...]
    audio: Path
    audio_duration: float
    audio_rate: int

    @property
    def ref(self) -> ClipInfo:
        return self.clips[0]

    def describe(self) -> str:
        r = self.ref
        return (f"{len(self.clips)} clips · {r.frames} frames @ {float(r.fps):g} fps "
                f"= {r.duration:.3f} s · {r.width}×{r.height}     "
                f"Mix: {fmt_time(self.audio_duration)} @ {self.audio_rate / 1000:g} kHz")


def ffprobe_json(args: list[str]) -> dict:
    proc = subprocess.run(["ffprobe", "-v", "error", "-of", "json", *args],
                          capture_output=True, text=True, creationflags=NO_WINDOW)
    if proc.returncode != 0:
        raise RenderError(f"ffprobe failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe_clip(path: Path) -> ClipInfo:
    data = ffprobe_json(["-select_streams", "v:0", "-show_entries",
                         "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
                         str(path)])
    if not data.get("streams"):
        raise RenderError(f"No video stream in {path.name}")
    s = data["streams"][0]
    rate = s.get("avg_frame_rate") or "0/0"
    if rate in ("0/0", "0"):
        rate = s["r_frame_rate"]
    frames = str(s.get("nb_frames", ""))
    if not frames.isdigit():  # container doesn't store it: count packets
        counted = ffprobe_json(["-select_streams", "v:0", "-count_packets",
                                "-show_entries", "stream=nb_read_packets", str(path)])
        frames = counted["streams"][0]["nb_read_packets"]
    return ClipInfo(path, int(s["width"]), int(s["height"]), Fraction(rate), int(frames))


def probe_audio(path: Path) -> tuple[float, int]:
    """Return (duration in seconds, sample rate) of the first audio stream."""
    data = ffprobe_json(["-select_streams", "a:0", "-show_entries",
                         "stream=duration,sample_rate:format=duration", str(path)])
    stream = (data.get("streams") or [{}])[0]
    if "sample_rate" not in stream:
        raise RenderError(f"No audio stream in {path.name}")
    for src in stream, data.get("format", {}):
        try:
            return float(src["duration"]), int(stream["sample_rate"])
        except (KeyError, TypeError, ValueError):
            continue
    raise RenderError(f"Could not read the duration of {path.name}")


def require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RenderError(f"{tool} was not found on PATH")


def find_audio(folder: Path) -> Path:
    """The single audio file in `folder` (the default DJ Mix)."""
    mixes = list_media(folder, AUDIO_EXTS)
    if len(mixes) != 1:
        names = ", ".join(p.name for p in mixes) or "none"
        raise RenderError(f"Expected exactly one audio file in {folder} (found: {names}). "
                          "Choose the mix explicitly.")
    return mixes[0]


def probe_inputs(video_dir: Path, audio: Path) -> Inputs:
    """Probe all clips (in parallel) and the mix; check the clips all match."""
    require_tools()
    clips = list_media(video_dir, VIDEO_EXTS)
    if not clips:
        raise RenderError(f"No video clips found in {video_dir}")
    if not audio.is_file():
        raise RenderError(f"DJ Mix not found: {audio}")

    with ThreadPoolExecutor(max_workers=min(8, len(clips))) as pool:
        infos = list(pool.map(probe_clip, clips))
    ref = infos[0]
    mismatched = [i for i in infos
                  if (i.frames, i.fps, i.width, i.height) != (ref.frames, ref.fps, ref.width, ref.height)]
    if mismatched:
        lines = "\n".join(f"  {i.path.name}: {i.frames} frames @ {float(i.fps):g} fps, {i.width}x{i.height}"
                          for i in [ref, *mismatched])
        raise RenderError("All clips must have the same length, frame rate and size:\n" + lines)

    duration, rate = probe_audio(audio)
    return Inputs(tuple(infos), audio, duration, rate)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_media(folder: Path, exts: set[str]) -> list[Path]:
    if not folder.is_dir():
        raise RenderError(f"Folder not found: {folder}")
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in exts), key=natural_key)


def even_ceil(x: float) -> int:
    return int(math.ceil(x / 2.0) * 2)


def cover_size(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    """Smallest even size that covers dst while keeping the source aspect ratio."""
    scale = max(dst_w / src_w, dst_h / src_h)
    return max(dst_w, even_ceil(src_w * scale)), max(dst_h, even_ceil(src_h * scale))


def fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}"


def concat_escape(path: Path) -> str:
    return path.resolve().as_posix().replace("'", r"'\''")


def check_bpm(value: float, what: str) -> None:
    if not 40 <= value <= 250:
        raise RenderError(f"{what} {value:g} looks wrong (expected roughly 122-126)")


def check_nvenc(encoder: str) -> None:
    """Fail early with a clear message if NVENC can't actually open."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=c=black:s=1280x720:r=24:d=0.2", "-c:v", encoder, "-f", "null", "-"],
        capture_output=True, text=True, creationflags=NO_WINDOW)
    if proc.returncode != 0:
        raise RenderError(f"{encoder} is not usable on this machine:\n{proc.stderr.strip()}\n"
                          "Update the NVIDIA driver, or close other apps using NVENC.")


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """User choices. Field names match the command-line options."""
    bpm: float
    target_bpm: float | None = None      # None = keep the mix's tempo
    stretch: str = "keylock"             # keylock | varispeed
    bars: int | None = None              # None = nearest whole number of bars
    beats_per_bar: int = 4
    offset: float = 0.0                  # first downbeat, in ORIGINAL mix time
    fps: float | None = None             # None = same as the clips
    width: int = 1920
    height: int = 1080
    fade: float = 4.0
    audio_fade: bool = False
    codec: str = "h264"                  # h264 | hevc
    cq: int = 19
    preset: str = "p7"
    audio_bitrate: str = "320k"
    cpu_decode: bool = False
    preview: float | None = None         # render only the first N seconds
    output: Path | None = None           # None = ./output/<auto name>.mp4
    overlay: Path | None = None          # transparent image shown centred over the intro
    overlay_size: float = 50.0           # image width as % of the video width
    overlay_start: float = 1.0           # seconds into the video
    overlay_duration: float = 5.0        # time on screen, including the fades
    overlay_fade_in: float = 0.5
    overlay_fade_out: float = 1.0


@dataclass
class Plan:
    inputs: Inputs
    settings: Settings
    src_bpm: float
    bpm: float                # effective output tempo
    ratio: float              # output tempo / mix tempo
    mix_dur: float            # mix length after the tempo change
    target_dur: float         # what gets rendered (mix_dur, or the preview length)
    fade: float
    bar: float
    bars: int
    segment: float            # seconds each clip occupies
    offset: float             # first downbeat in output time
    shift: float
    out_fps: Fraction
    playlist: list[Path]
    scaled: tuple[int, int]
    audio_chain: list[str]
    output: Path

    @property
    def retempo(self) -> bool:
        return abs(self.ratio - 1) > 1e-9

    @property
    def speed(self) -> float:
        return self.inputs.ref.duration / self.segment

    @property
    def frame_step(self) -> float:
        return self.segment / self.inputs.ref.frames

    @property
    def encoder(self) -> str:
        return f"{self.settings.codec}_nvenc"

    @property
    def partial_output(self) -> Path:
        return self.output.with_name(self.output.stem + ".partial" + self.output.suffix)

    @property
    def overlay_window(self) -> tuple[float, float, float, float] | None:
        """(start, end, fade_in, fade_out) of the overlay, or None if it's off."""
        s = self.settings
        if s.overlay is None:
            return None
        start, dur = max(0.0, s.overlay_start), s.overlay_duration
        fin, fout = max(0.0, s.overlay_fade_in), max(0.0, s.overlay_fade_out)
        if fin + fout > dur:   # fades longer than the time on screen: shrink them to fit
            k = dur / (fin + fout)
            fin, fout = fin * k, fout * k
        return start, start + dur, fin, fout

    def summary(self) -> list[str]:
        s, ref = self.settings, self.inputs.ref
        lines = [f"  DJ Mix      : {self.inputs.audio.name}  "
                 f"({fmt_time(self.inputs.audio_duration)} @ {self.src_bpm:g} BPM)"]
        if self.retempo:
            how = (f"varispeed, pitch {12 * math.log2(self.ratio):+.2f} semitones"
                   if s.stretch == "varispeed" else "keylock, original pitch")
            lines.append(f"  Re-tempo    : {self.src_bpm:g} -> {round(self.bpm, 4):g} BPM  "
                         f"(x{self.ratio:.5f}, {how}), new length {fmt_time(self.mix_dur)}")
        lines += [
            f"  Tempo       : {round(self.bpm, 4):g} BPM  ->  1 bar = {self.bar:.4f} s",
            f"  Clips       : {len(self.inputs.clips)} x {ref.frames} frames @ {float(ref.fps):g} fps "
            f"= {ref.duration:.4f} s, {ref.width}x{ref.height}",
            f"  Each clip   : {ref.duration:.4f} s -> {self.segment:.4f} s ({self.bars} bars), "
            f"speed x{self.speed:.4f} ({(self.speed - 1) * 100:+.2f}%)",
            f"  Output      : {s.width}x{s.height} @ {float(self.out_fps):g} fps, "
            f"~{self.segment * float(self.out_fps):.2f} frames per clip, {s.codec.upper()} NVENC "
            f"CQ {s.cq} {s.preset} + AAC {s.audio_bitrate}",
            f"  Sequence    : {len(self.playlist)} clip plays ({len(self.playlist) / len(self.inputs.clips):.1f} "
            f"loops of {len(self.inputs.clips)}), cut to {fmt_time(self.target_dur)}"
            + (f", {self.fade:g} s fade to black" + (" + audio fade" if s.audio_fade else "")
               if self.fade else ""),
        ]
        if s.offset:
            lines.append(f"  Downbeat    : first clip cut at {self.offset:.3f} s")
        if self.overlay_window:
            start, end, fin, fout = self.overlay_window
            lines.append(f"  Overlay     : {Path(s.overlay).name}, centred at {s.overlay_size:g}% width, "
                         f"{start:g}-{end:g} s (fade in {fin:g} s, out {fout:g} s)")
        if s.preview:
            lines.append(f"  PREVIEW     : first {self.target_dur:g} s only")
        lines.append(f"  File        : {self.output}")
        return lines

    def concat_text(self) -> str:
        return "ffconcat version 1.0\n" + "".join(f"file '{concat_escape(p)}'\n" for p in self.playlist)

    def command(self, list_file: Path, output: Path | None = None) -> list[str]:
        s = self.settings
        output = output or self.output
        sw, sh = self.scaled

        # settb=1/1e6 -> timestamps in microseconds; frame N of the whole sequence
        # is placed at N * frame_step. Computed from N (not from accumulated
        # durations), so the picture stays locked to the beat for the full mix.
        timing = (f"settb=1/1000000,"
                  f"setpts='N*{self.frame_step * 1e6:.6f}-{self.shift * 1e6:.3f}'")
        if self.shift > 0:
            timing += ",trim=start=0"
        timing += f",fps=fps={self.out_fps}:start_time=0"

        crop = f"crop={s.width}:{s.height}"
        fade_f = (f"fade=t=out:st={self.target_dur - self.fade:.6f}:d={self.fade:.6f}:color=black,"
                  if self.fade else "")
        if s.cpu_decode:
            base = f"[0:v]{timing},scale={sw}:{sh}:flags=lanczos,{crop}"
            hw_in: list[str] = []
        else:
            # NVDEC decodes, but hands frames back in system memory; they're uploaded
            # again for scale_cuda. Keeping decoder frames on the GPU would make
            # ffmpeg rebuild the filtergraph whenever a clip's stream parameters
            # differ (new GPU frame pool), which resets N and breaks the timing.
            base = (f"[0:v]{timing},hwupload_cuda,scale_cuda=w={sw}:h={sh}:interp_algo=lanczos,"
                    f"hwdownload,format=nv12,{crop}")
            hw_in = ["-hwaccel", "cuda"]

        overlay_in: list[str] = []
        if self.overlay_window:
            # The still image is looped into a short stream that ends with the overlay;
            # its alpha is faded in/out, and the overlay filter is only enabled during
            # that window, so the rest of the video passes straight through.
            start, end, fin, fout = self.overlay_window
            overlay_in = ["-loop", "1", "-framerate", str(self.out_fps), "-t", f"{end:.6f}",
                          "-i", str(s.overlay)]
            box_w = round(s.width * s.overlay_size / 100)
            box_h = round(s.height * 0.9)
            logo = (f"[2:v]format=rgba,scale=w={box_w}:h={box_h}:"
                    f"force_original_aspect_ratio=decrease:flags=lanczos")
            if fin > 0:
                logo += f",fade=t=in:st={start:.6f}:d={fin:.6f}:alpha=1"
            if fout > 0:
                logo += f",fade=t=out:st={end - fout:.6f}:d={fout:.6f}:alpha=1"
            vf = (f"{base}[base];{logo}[logo];"
                  f"[base][logo]overlay=x=(W-w)/2:y=(H-h)/2:eof_action=pass:"
                  f"enable='between(t,{start:.6f},{end:.6f})',{fade_f}format=yuv420p[v]")
        else:
            vf = f"{base},{fade_f}format=yuv420p[v]"

        gop = str(max(1, round(float(self.out_fps) * 2)))   # keyframe every 2 s
        venc = ["-c:v", self.encoder, "-preset", s.preset, "-tune", "hq",
                "-rc", "vbr", "-cq", str(s.cq), "-b:v", "0",
                "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32",
                "-bf", "3", "-g", gop]
        venc += ["-profile:v", "high"] if s.codec == "h264" else ["-profile:v", "main", "-tag:v", "hvc1"]

        audio_chain = list(self.audio_chain)
        if s.audio_fade and self.fade:
            audio_chain.append(f"afade=t=out:st={self.target_dur - self.fade:.6f}:d={self.fade:.6f}")
        # -af runs as its own filtergraph (own thread), so rubberband doesn't slow the video.
        aenc = ["-c:a", "aac", "-b:a", s.audio_bitrate, "-ar", "48000"]
        if audio_chain:
            aenc += ["-af", ",".join(audio_chain)]

        # -reinit_filter 0: never rebuild the filtergraph when a clip's metadata
        # (colour tags etc.) differs from the previous one. A rebuild would reset
        # setpts' frame counter N and throw the whole sequence off the beat.
        # Clip size and frame rate are already checked to be identical.
        return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-progress", "pipe:1", "-nostats",
                *hw_in, "-reinit_filter", "0",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-i", str(self.inputs.audio),
                *overlay_in,
                "-filter_complex", vf,
                "-map", "[v]", "-map", "1:a:0",          # dancer-clip audio is never mapped
                *venc, *aenc,
                "-t", f"{self.target_dur:.6f}",
                "-movflags", "+faststart",
                "-y", str(output)]


def make_plan(inputs: Inputs, s: Settings, base_dir: Path = HERE) -> Plan:
    """Work out all timing for a render. Cheap: no file access."""
    src_bpm = s.bpm
    check_bpm(src_bpm, "BPM")
    target_bpm = s.target_bpm if s.target_bpm is not None else src_bpm
    check_bpm(target_bpm, "Target BPM")
    if s.bars is not None and s.bars < 1:
        raise RenderError("Bars per clip must be at least 1")
    if s.beats_per_bar < 1:
        raise RenderError("Beats per bar must be at least 1")
    if s.overlay is not None:
        if not Path(s.overlay).is_file():
            raise RenderError(f"Overlay image not found: {s.overlay}")
        if s.overlay_duration <= 0:
            raise RenderError("Overlay duration must be more than 0 seconds")
        if not 1 <= s.overlay_size <= 100:
            raise RenderError("Overlay size must be between 1 and 100% of the video width")

    # ---- audio tempo change -------------------------------------------------
    ratio = target_bpm / src_bpm              # >1 = faster
    audio_chain: list[str] = []
    if abs(ratio - 1) > 1e-9:
        if s.stretch == "varispeed":
            # Re-label the sample rate (speeds up/slows down, pitch follows), then
            # resample back. asetrate needs a whole number of Hz, so take the exact
            # ratio that gives and lock the video to that tempo instead.
            new_rate = round(inputs.audio_rate * ratio)
            ratio = new_rate / inputs.audio_rate
            target_bpm = src_bpm * ratio
            audio_chain.append(f"asetrate={new_rate}")
        else:
            audio_chain.append(f"rubberband=tempo={ratio:.12f}:channels=together")
        audio_chain.append(SOXR)
    elif inputs.audio_rate != 48000:
        audio_chain.append(SOXR)
    mix_dur = inputs.audio_duration / ratio

    # ---- video timing -------------------------------------------------------
    ref = inputs.ref
    out_fps = Fraction(s.fps).limit_denominator(1001) if s.fps else ref.fps
    bar = s.beats_per_bar * 60.0 / target_bpm
    bars = s.bars or max(1, round(ref.duration / bar))
    segment = bars * bar

    target_dur = min(mix_dur, s.preview) if s.preview else mix_dur
    fade = min(max(s.fade, 0.0), target_dur)

    # Shift the clip grid so that a clip boundary lands on the first downbeat.
    offset = s.offset / ratio
    shift = (segment - (offset % segment)) % segment
    n_entries = math.ceil((target_dur + shift) / segment) + 1
    playlist = [inputs.clips[i % len(inputs.clips)].path for i in range(n_entries)]

    # ---- output path --------------------------------------------------------
    if s.output is None:
        tempo = (f"{src_bpm:g}bpm" if abs(ratio - 1) <= 1e-9
                 else f"{src_bpm:g}to{round(target_bpm, 2):g}bpm_{s.stretch}")
        output = base_dir / "output" / f"{inputs.audio.stem}_{tempo}.mp4"
    else:
        output = Path(s.output).with_suffix(".mp4")
    if s.preview:   # never let a preview overwrite the real render
        output = output.with_name(f"{output.stem}_preview{int(target_dur)}s.mp4")

    return Plan(inputs=inputs, settings=s, src_bpm=src_bpm, bpm=target_bpm, ratio=ratio,
                mix_dur=mix_dur, target_dur=target_dur, fade=fade, bar=bar, bars=bars,
                segment=segment, offset=offset, shift=shift, out_fps=out_fps, playlist=playlist,
                scaled=cover_size(ref.width, ref.height, s.width, s.height),
                audio_chain=audio_chain, output=output)


def finish_output(plan: Plan, success: bool) -> None:
    """Rename the finished .partial file into place, or remove it."""
    if success:
        os.replace(plan.partial_output, plan.output)
    else:
        plan.partial_output.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Loop beat-synced dancer clips over a DJ Mix and render a 1080p MP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--bpm", type=float, help="tempo of the DJ Mix as recorded (prompted if omitted)")
    ap.add_argument("--target-bpm", type=float,
                    help="tempo of the output video; the mix is time-stretched to it "
                         "(default: same as --bpm, i.e. audio untouched)")
    ap.add_argument("--stretch", choices=["keylock", "varispeed"], default="keylock",
                    help="how to change the mix's tempo: keylock = keep the original pitch "
                         "(rubberband), varispeed = pitch follows tempo like a turntable (soxr)")
    ap.add_argument("--video-dir", type=Path, default=HERE / "video")
    ap.add_argument("--audio", type=Path,
                    help="the DJ Mix file (default: the only audio file in ./audio)")
    ap.add_argument("-o", "--output", type=Path,
                    help="output .mp4 (default: ./output/<mix>_<bpm>bpm.mp4; "
                         "previews get a _preview<N>s suffix)")
    ap.add_argument("--bars", type=int,
                    help="bars per clip (default: whole number of bars closest to the clip length)")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--offset", type=float, default=0.0,
                    help="time of the first downbeat in the ORIGINAL mix, in seconds "
                         "(clip cuts are aligned to it)")
    ap.add_argument("--fps", type=float, help="output frame rate (default: same as the clips)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fade", type=float, default=4.0, help="fade-to-black length at the end (0 = off)")
    ap.add_argument("--audio-fade", action="store_true", help="also fade the audio out over --fade seconds")
    ap.add_argument("--codec", choices=["h264", "hevc"], default="h264")
    ap.add_argument("--cq", type=int, default=19, help="NVENC constant quality (lower = better/bigger)")
    ap.add_argument("--preset", default="p7", help="NVENC preset p1 (fastest) .. p7 (best)")
    ap.add_argument("--audio-bitrate", default="320k", help="AAC bitrate")
    ap.add_argument("--cpu-decode", action="store_true",
                    help="decode and scale on the CPU (encoding still uses NVENC)")
    ap.add_argument("--preview", type=float, metavar="SECONDS",
                    help="render only the first SECONDS (quick test of sync and quality)")
    ov = ap.add_argument_group("logo overlay (e.g. a transparent subscribe.png over the intro)")
    ov.add_argument("--overlay", type=Path, metavar="IMAGE",
                    help="transparent image to show centred over the start of the video")
    ov.add_argument("--overlay-size", type=float, default=50.0, metavar="PERCENT",
                    help="image width as a percentage of the video width")
    ov.add_argument("--overlay-start", type=float, default=1.0, metavar="S",
                    help="when the image appears (seconds into the video)")
    ov.add_argument("--overlay-duration", type=float, default=5.0, metavar="S",
                    help="how long it is on screen, including the fades")
    ov.add_argument("--overlay-fade-in", type=float, default=0.5, metavar="S")
    ov.add_argument("--overlay-fade-out", type=float, default=1.0, metavar="S")
    ap.add_argument("-y", "--overwrite", action="store_true", help="overwrite the output file")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and ffmpeg command, then exit")
    return ap.parse_args()


def ask_bpm(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number, e.g. 124 or 123.5")


def main() -> int:
    args = parse_args()
    require_tools()
    audio = args.audio or find_audio(HERE / "audio")

    if args.bpm is None:
        args.bpm = ask_bpm("DJ Mix BPM (usually 122-126): ")
        if args.target_bpm is None:
            args.target_bpm = ask_bpm(f"Output BPM [Enter = keep {args.bpm:g}]: ", default=args.bpm)

    print("Probing clips ...")
    inputs = probe_inputs(args.video_dir, audio)
    settings = Settings(**{f.name: getattr(args, f.name) for f in fields(Settings)})
    plan = make_plan(inputs, settings)
    print("\n" + "\n".join(plan.summary()))

    with tempfile.TemporaryDirectory(prefix="dancesync_") as tmp:
        list_file = Path(tmp) / "clips.txt"
        list_file.write_text(plan.concat_text(), encoding="utf-8")

        if args.dry_run:
            print("\n" + subprocess.list2cmdline(plan.command(list_file)))
            return 0

        if plan.output.exists() and not args.overwrite:
            if not sys.stdin.isatty() or input(f"\n{plan.output.name} exists. Overwrite? [y/N] ") \
                    .strip().lower() not in ("y", "yes"):
                raise RenderError("Not overwriting (use -y to force).")
        plan.output.parent.mkdir(parents=True, exist_ok=True)

        check_nvenc(plan.encoder)
        print(f"\nRendering -> {plan.output}\n")
        code = run_ffmpeg(plan.command(list_file, plan.partial_output), plan.target_dur)
        finish_output(plan, code == 0)
        return code


def run_ffmpeg(cmd: list[str], total: float) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    bar = (tqdm(total=round(total, 1), unit="s", unit_scale=False,
                bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}] {postfix}")
           if tqdm else None)
    done = 0.0
    stats: dict[str, str] = {}
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            key, _, val = line.strip().partition("=")
            stats[key] = val
            if key == "out_time_us" and val.lstrip("-").isdigit():
                done = min(total, max(0, int(val)) / 1e6)
            elif key == "progress":
                if bar:
                    bar.n = round(done, 1)
                    bar.set_postfix_str(f"{stats.get('fps', '?')} fps, {stats.get('speed', '?').strip()}")
                    bar.refresh()
                else:
                    print(f"\r  {fmt_time(done)} / {fmt_time(total)}  "
                          f"({done / total * 100:5.1f}%)  {stats.get('speed', '')}   ", end="", flush=True)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print("\nCancelled.")
        return 130
    finally:
        if bar:
            bar.close()
    if code != 0:
        print(f"\nffmpeg exited with code {code}", file=sys.stderr)
    else:
        print("\nDone.")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RenderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
