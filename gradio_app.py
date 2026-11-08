#!/usr/bin/env python3
"""
Web UI for the beat-synced dance video builder (Gradio).

Runs anywhere Python and ffmpeg do, and is made for Google Colab
(open VisualSync_Colab.ipynb there). Probing, planning and the ffmpeg command
all come from make_dance_video, so the web UI, the desktop GUI and the command
line render identically.

    pip install gradio
    python gradio_app.py              # http://127.0.0.1:7860
    python gradio_app.py --share      # also a public *.gradio.live link (Colab)

On a machine without a working NVIDIA GPU it falls back to CPU decoding and
encoding (libx264/libx265) automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr

import make_dance_video as mdv

HERE = mdv.HERE
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir()
DEFAULT_LOGO = HERE / "subscribe.png"
DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive/VisualSync"
STAGE_ROOT = Path(tempfile.gettempdir()) / "visualsync_uploads"
# Local-disk working area on Colab: renders and cached copies of Drive clips.
LOCAL_ROOT = Path("/content/visualsync_local") if ON_COLAB else Path(tempfile.gettempdir()) / "visualsync_local"
RENDER_DIR = LOCAL_ROOT / "render"
CLIP_CACHE = LOCAL_ROOT / "clips"
CLIP_CACHE_MAX = 20e9              # don't cache clip sets bigger than this
DEFAULT_CQ = {"hevc": 24, "h264": 21}

RESOLUTIONS = [("720p (1280×720)", "1280x720"), ("1080p (1920×1080)", "1920x1080"),
               ("1440p (2560×1440)", "2560x1440"), ("4K (3840×2160)", "3840x2160")]
CODECS = [("H.264 – most compatible", "h264"), ("HEVC / H.265 – smaller files", "hevc")]
STRETCHES = [("Keylock – keep original pitch", "keylock"),
             ("Varispeed – pitch follows tempo", "varispeed")]
PRESETS = [(f"p{i}" + {7: " – best quality", 4: " – balanced", 1: " – fastest"}.get(i, ""), f"p{i}")
           for i in range(7, 0, -1)]
AUDIO_BITRATES = [("320 kbps", "320k"), ("256 kbps", "256k"), ("192 kbps", "192k")]
HW_CHOICES = [("Auto", "auto"), ("GPU (NVIDIA)", "gpu"), ("CPU", "cpu")]
UPLOAD, PATHS = "upload", "paths"

JOB: dict = {}                      # the running render: {"proc": Popen, "cancel": bool}
RENDER_LOCK = threading.Lock()      # one render at a time
_probe_cache: dict = {}


# --------------------------------------------------------------------------- #
# What this machine's ffmpeg can do
# --------------------------------------------------------------------------- #
def _ffmpeg_ok(args: list[str]) -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
                              capture_output=True, text=True, timeout=60)
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


_hw: dict | None = None


def detect_hardware() -> dict:
    """Test (once) which GPU and ffmpeg features actually work here."""
    global _hw
    if _hw is not None:
        return _hw
    hw = {"ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")), "gpu": "",
          "version": "", "rubberband": False, "soxr": False, "cuda_scale": False,
          "h264_nvenc": False, "hevc_nvenc": False, "libx264": False, "libx265": False}
    if not hw["ffmpeg"]:
        _hw = hw
        return hw
    try:
        hw["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        hw["version"] = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True,
                                       text=True).stdout.splitlines()[0]
        filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True,
                                 text=True).stdout
        hw["rubberband"] = " rubberband " in filters
    except (OSError, IndexError):
        pass
    video = ["-f", "lavfi", "-i", "color=c=black:s=1280x720:r=24:d=0.2"]
    for enc in ("h264_nvenc", "hevc_nvenc", "libx264", "libx265"):
        hw[enc] = _ffmpeg_ok([*video, "-c:v", enc, "-f", "null", "-"])
    # The exact GPU scaling chain make_dance_video uses (needs ffmpeg 5+ for interp_algo).
    hw["cuda_scale"] = bool(hw["gpu"]) and _ffmpeg_ok(
        [*video, "-vf", "format=nv12,hwupload_cuda,scale_cuda=w=640:h=360:interp_algo=lanczos,"
                        "hwdownload,format=nv12", "-f", "null", "-"])
    hw["soxr"] = _ffmpeg_ok(["-f", "lavfi", "-i", "anullsrc=r=44100:d=0.1",
                             "-af", mdv.SOXR, "-f", "null", "-"])
    if not hw["soxr"]:   # fall back to ffmpeg's own resampler
        mdv.SOXR = "aresample=48000"
    _hw = hw
    return hw


def hardware_markdown(hw: dict) -> str:
    if not hw["ffmpeg"]:
        return "⚠️ **ffmpeg / ffprobe not found.** Install ffmpeg and restart."
    yes = lambda ok: "✅" if ok else "❌"   # noqa: E731
    gpu = hw["gpu"] or "none"
    lines = [f"**GPU:** {gpu} · **ffmpeg:** {hw['version'].replace('ffmpeg version ', '')[:40]}",
             f"{yes(hw['h264_nvenc'])} NVENC H.264 · {yes(hw['hevc_nvenc'])} NVENC HEVC · "
             f"{yes(hw['cuda_scale'])} GPU scaling · {yes(hw['libx264'])} CPU H.264 · "
             f"{yes(hw['libx265'])} CPU HEVC · {yes(hw['rubberband'])} keylock (rubberband)"]
    if hw["gpu"] and not hw["h264_nvenc"]:
        lines.append("ℹ️ This GPU can't encode video (A100/H100 have no NVENC). "
                     "Auto uses the CPU — for speed, pick a **T4** or **L4** runtime.")
    elif not hw["gpu"]:
        lines.append("ℹ️ No NVIDIA GPU: rendering runs on the CPU. A faster preset (p3–p4) "
                     "keeps long mixes manageable.")
    return "  \n".join(lines)


def resolve_hw(encode: str, decode: str, codec: str, hw: dict) -> tuple[bool, bool]:
    """(cpu_encode, cpu_decode) for the chosen Auto/GPU/CPU options."""
    cpu_encode = {"gpu": False, "cpu": True}.get(encode, not hw[f"{codec}_nvenc"])
    cpu_decode = {"gpu": False, "cpu": True}.get(decode, not hw["cuda_scale"])
    return cpu_encode, cpu_decode


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def _file_path(f) -> Path:
    return Path(f if isinstance(f, (str, os.PathLike)) else f.name)


def stage_uploads(files) -> Path:
    """Gather uploaded clips (each in its own temp folder) into one folder of links."""
    paths = [_file_path(f) for f in files]
    paths = [p for p in paths if p.suffix.lower() in mdv.VIDEO_EXTS]
    if not paths:
        raise mdv.RenderError("None of the uploaded files is a video clip "
                              f"({', '.join(sorted(mdv.VIDEO_EXTS))}).")
    key = hashlib.sha1("\n".join(sorted(map(str, paths))).encode()).hexdigest()[:12]
    folder = STAGE_ROOT / key
    if not folder.is_dir():
        tmp = STAGE_ROOT / (key + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        used: set[str] = set()
        for p in paths:
            name, n = p.name, 2
            while name.lower() in used:
                name, n = f"{p.stem}_{n}{p.suffix}", n + 1
            used.add(name.lower())
            try:
                (tmp / name).symlink_to(p.resolve())
            except OSError:
                shutil.copy2(p, tmp / name)
        tmp.rename(folder)
    return folder


def resolve_sources(mode, clip_files, mix_file, clips_dir, mix_path) -> tuple[Path, Path]:
    if mode == UPLOAD:
        if not clip_files:
            raise mdv.RenderError("Upload the dancer clips.")
        if not mix_file:
            raise mdv.RenderError("Upload the DJ Mix.")
        return stage_uploads(clip_files), _file_path(mix_file)
    if not (clips_dir or "").strip():
        raise mdv.RenderError("Enter the dancer clips folder.")
    if not (mix_path or "").strip():
        raise mdv.RenderError("Enter the DJ Mix file (or a folder holding only the mix).")
    video_dir = Path(clips_dir.strip()).expanduser()
    mix = Path(mix_path.strip()).expanduser()
    return video_dir, (mdv.find_audio(mix) if mix.is_dir() else mix)


def probe(video_dir: Path, audio: Path) -> mdv.Inputs:
    """mdv.probe_inputs, cached while the files are unchanged."""
    clips = mdv.list_media(video_dir, mdv.VIDEO_EXTS)
    try:
        sig = tuple((p.name, p.stat().st_size, p.stat().st_mtime) for p in clips)
        key = (str(video_dir), sig, str(audio), audio.stat().st_mtime)
    except OSError:
        key = None
    if key is None or key not in _probe_cache:
        inputs = mdv.probe_inputs(video_dir, audio)
        if key is None:
            return inputs
        _probe_cache[key] = inputs
    return _probe_cache[key]


# --------------------------------------------------------------------------- #
# Google Drive: keep the heavy I/O on Colab's local disk
# --------------------------------------------------------------------------- #
def on_drive(path: Path) -> bool:
    """True for files on the mounted Google Drive (a slow network filesystem)."""
    try:
        return Path(path).resolve().is_relative_to(DRIVE_MOUNT)
    except OSError:
        return False


def free_bytes(folder: Path) -> float:
    folder.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(folder).free


def use_local_render(plan: mdv.Plan) -> None:
    """Render a Drive-bound video on local disk; finish_output copies it over at the end.

    ffmpeg's +faststart rewrites the whole file once encoding is done (to move the
    index to the front), which is slow on the Drive mount. Skipped if the local disk
    is short of space (estimate: 20 Mbit/s for the whole video, plus 2 GB headroom).
    """
    if not on_drive(plan.output):
        return
    try:
        if free_bytes(RENDER_DIR) > plan.target_dur * 2.5e6 + 2e9:
            plan.work_dir = RENDER_DIR
    except OSError:
        pass


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def cache_drive_clips(plan: mdv.Plan) -> int:
    """Copy the dancer clips from Drive to local disk once, and point the plan at them.

    The concat list re-opens every clip many times over a long mix; reading local
    copies avoids Drive latency on each one. Copies are reused by later renders
    (keyed on path, size and modification time). Returns how many clips were copied.
    """
    clips = [c.path for c in plan.inputs.clips if on_drive(c.path)]
    if not clips:
        return 0
    try:
        stats = {p: p.stat() for p in clips}
        total = sum(st.st_size for st in stats.values())
        if total > CLIP_CACHE_MAX or free_bytes(CLIP_CACHE) < total + 5e9:
            return 0
    except OSError:
        return 0

    mapping: dict[Path, Path] = {}
    todo: list[tuple[Path, Path]] = []
    for p, st in stats.items():
        key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
        dst = CLIP_CACHE / f"{key}{p.suffix.lower()}"
        mapping[p] = dst
        if not (dst.is_file() and dst.stat().st_size == st.st_size):
            todo.append((p, dst))
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:   # Drive reads are latency-bound
            list(pool.map(lambda job: _copy_atomic(*job), todo))
    plan.playlist = [mapping.get(p, p) for p in plan.playlist]
    return len(todo)


# --------------------------------------------------------------------------- #
# Settings -> plan
# --------------------------------------------------------------------------- #
FIELDS = ["mode", "clip_files", "mix_file", "clips_dir", "mix_path", "output_dir",
          "bpm", "retempo", "target_bpm", "stretch", "bars", "beats_per_bar", "offset",
          "resolution", "codec", "cq", "preset", "audio_bitrate", "fade", "audio_fade", "fps",
          "encode", "decode",
          "overlay_on", "overlay", "overlay_size", "overlay_start", "overlay_duration",
          "overlay_fade_in", "overlay_fade_out", "preview_len"]


def num(value, default: float = 0.0) -> float:
    try:
        return default if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return default


def make_plan(vals: tuple, preview: bool) -> tuple[mdv.Plan, dict]:
    v = dict(zip(FIELDS, vals))
    hw = detect_hardware()
    if not hw["ffmpeg"]:
        raise mdv.RenderError("ffmpeg / ffprobe not found.")
    video_dir, audio = resolve_sources(v["mode"], v["clip_files"], v["mix_file"],
                                       v["clips_dir"], v["mix_path"])
    inputs = probe(video_dir, audio)

    retempo = bool(v["retempo"])
    if retempo and v["stretch"] == "keylock" and not hw["rubberband"]:
        raise mdv.RenderError("This ffmpeg has no rubberband filter, so keylock isn't available. "
                              "Choose Varispeed, or keep the mix's tempo.")
    overlay = None
    if v["overlay_on"]:
        if not v["overlay"]:
            raise mdv.RenderError("Choose a logo image, or untick the logo overlay.")
        overlay = _file_path(v["overlay"])
    cpu_encode, cpu_decode = resolve_hw(v["encode"], v["decode"], v["codec"], hw)
    if cpu_encode and not hw["libx265" if v["codec"] == "hevc" else "libx264"]:
        raise mdv.RenderError("This ffmpeg has no CPU encoder for that codec. Try H.264.")
    width, height = (int(x) for x in v["resolution"].split("x"))

    settings = mdv.Settings(
        bpm=num(v["bpm"], 124),
        target_bpm=num(v["target_bpm"], num(v["bpm"], 124)) if retempo else None,
        stretch=v["stretch"],
        bars=int(num(v["bars"])) or None,
        beats_per_bar=int(num(v["beats_per_bar"], 4)),
        offset=num(v["offset"]),
        fps=num(v["fps"]) or None,
        width=width, height=height,
        fade=num(v["fade"], 4.0),
        audio_fade=bool(v["audio_fade"]),
        codec=v["codec"],
        cq=int(num(v["cq"], DEFAULT_CQ.get(v["codec"], 24))),
        preset=v["preset"],
        audio_bitrate=v["audio_bitrate"],
        cpu_decode=cpu_decode,
        cpu_encode=cpu_encode,
        preview=max(1.0, num(v["preview_len"], 30)) if preview else None,
        overlay=overlay,
        overlay_size=num(v["overlay_size"], 70),
        overlay_start=num(v["overlay_start"], 1.0),
        overlay_duration=num(v["overlay_duration"], 5.0),
        overlay_fade_in=num(v["overlay_fade_in"], 0.5),
        overlay_fade_out=num(v["overlay_fade_out"], 1.0),
    )
    plan = mdv.make_plan(inputs, settings, base_dir=HERE)
    out_dir = (v["output_dir"] or "").strip()
    if out_dir:
        plan.output = Path(out_dir).expanduser() / plan.output.name
    use_local_render(plan)
    return plan, hw


def plan_text(plan: mdv.Plan) -> str:
    s = plan.settings
    where = (f"  Processing  : decode+scale on {'CPU' if s.cpu_decode else 'GPU'}, "
             f"encode on {'CPU' if s.cpu_encode else 'GPU (NVENC)'}")
    lines = [plan.inputs.describe(), "", *plan.summary(), where]
    if plan.work_dir:
        lines.append("  Rendered on : Colab's local disk, then copied to Drive")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Status display
# --------------------------------------------------------------------------- #
def hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def status_html(text: str, pct: float | None = None, kind: str = "info") -> str:
    colour = {"info": "inherit", "ok": "#1a9c4a", "error": "#d03b3b"}[kind]
    bar = (f'<progress value="{pct:.1f}" max="100" style="width:100%;height:14px"></progress>'
           if pct is not None else "")
    body = html.escape(text).replace("\n", "<br>")
    return f'<div style="color:{colour};line-height:1.5">{bar}<div>{body}</div></div>'


def downloadable(path: Path) -> bool:
    """Gradio only serves files from the folders passed to launch(allowed_paths)."""
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved.is_relative_to(root) for root in ALLOWED_ROOTS)


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def check_plan(*vals):
    try:
        plan, hw = make_plan(vals, preview=False)
        return status_html("Plan ready.", kind="ok"), plan_text(plan)
    except mdv.RenderError as exc:
        return status_html(str(exc), kind="error"), ""
    except Exception:
        return status_html("Unexpected error – see the Plan tab.", kind="error"), traceback.format_exc()


def _render(preview: bool, vals: tuple):
    """Generator: yields (status, plan, log, video, file) while ffmpeg runs."""
    nothing = (None, None)
    if not RENDER_LOCK.acquire(blocking=False):
        yield status_html("A render is already running.", kind="error"), gr.update(), gr.update(), *nothing
        return
    proc = None
    plan = None
    try:
        try:
            plan, hw = make_plan(vals, preview)
        except mdv.RenderError as exc:
            yield status_html(str(exc), kind="error"), "", "", *nothing
            return
        summary = plan_text(plan)
        yield status_html("Checking the encoder…", 0), summary, "", *nothing
        try:
            plan.output.parent.mkdir(parents=True, exist_ok=True)
            if plan.work_dir:
                plan.work_dir.mkdir(parents=True, exist_ok=True)
            note = mdv.check_plan_encoder(plan)
        except (mdv.RenderError, OSError) as exc:
            yield status_html(str(exc), kind="error"), summary, "", *nothing
            return

        if any(on_drive(c.path) for c in plan.inputs.clips):
            yield status_html("Copying the clips from Google Drive to local disk "
                              "(once; later renders reuse them)…", 0), summary, "", *nothing
            try:
                cache_drive_clips(plan)
            except OSError as exc:   # not fatal: read them from Drive instead
                note = ((note or "") + f"\nCouldn't cache the clips locally ({exc}); "
                        "reading them from Drive.").strip()

        with tempfile.TemporaryDirectory(prefix="dancesync_") as tmp:
            list_file = Path(tmp) / "clips.txt"
            list_file.write_text(plan.concat_text(), encoding="utf-8")
            cmd = plan.command(list_file, plan.partial_output)
            log = (note + "\n\n" if note else "") + subprocess.list2cmdline(cmd) + "\n"
            err_path = Path(tmp) / "ffmpeg_stderr.txt"
            total = plan.target_dur
            kind = f"preview ({total:g} s)" if preview else "full video"

            with open(err_path, "w+", encoding="utf-8", errors="replace") as err:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True,
                                        bufsize=1, cwd=HERE)
                JOB.update(proc=proc, cancel=False)
                t0 = time.monotonic()
                last = 0.0
                done = 0.0
                stats: dict[str, str] = {}
                yield status_html(f"Rendering {kind} → {plan.output.name}", 0), summary, log, *nothing
                for line in proc.stdout:  # type: ignore[union-attr]
                    key, _, val = line.strip().partition("=")
                    stats[key] = val
                    if key == "out_time_us" and val.lstrip("-").isdigit():
                        done = min(total, max(0, int(val)) / 1e6)
                    if key != "progress":
                        continue
                    now = time.monotonic()
                    if now - last < 1.0 and val != "end":
                        continue
                    last = now
                    elapsed = now - t0
                    pct = done / total * 100 if total else 0
                    eta = elapsed / done * (total - done) if done > 0 else 0
                    speed = stats.get("speed", "").strip()
                    text = (f"Rendering {kind}: {hms(done)} / {hms(total)} ({pct:.1f}%)  ·  "
                            f"{stats.get('fps', '?')} fps  ·  {speed or '?'}  ·  "
                            f"elapsed {hms(elapsed)}  ·  ETA {hms(eta)}")
                    yield status_html(text, pct), summary, log, *nothing
                code = proc.wait()
                err.seek(0)
                err_text = err.read().strip()

            if err_text:
                log += "\n" + err_text
            cancelled = JOB.get("cancel", False)
            ok = code == 0 and not cancelled
            if ok and plan.work_dir:
                yield status_html(f"Copying {plan.output.name} to Google Drive…", 100), \
                    summary, log, *nothing
            try:
                mdv.finish_output(plan, ok)
            except OSError as exc:
                yield status_html(f"Rendered, but copying to {plan.output.parent} failed ({exc}).\n"
                                  f"The video is still on Colab's disk: {plan.partial_output}",
                                  kind="error"), summary, log, *nothing
                return
            elapsed = hms(time.monotonic() - t0)
            if cancelled:
                yield status_html("Stopped. The unfinished file was removed.", kind="error"), \
                    summary, log, *nothing
            elif not ok:
                yield status_html(f"ffmpeg failed (exit code {code}) – see the ffmpeg log tab.",
                                  kind="error"), summary, log, *nothing
            else:
                size = plan.output.stat().st_size / 1e6
                msg = f"Done in {elapsed}: {plan.output}  ({size:,.0f} MB)"
                if downloadable(plan.output):
                    video = str(plan.output) if preview else None
                    yield status_html(msg, 100, kind="ok"), summary, log, video, str(plan.output)
                else:
                    yield status_html(msg + "\n(Saved outside the app's folders, so it isn't "
                                      "offered as a download here.)", 100, kind="ok"), \
                        summary, log, *nothing
    except Exception:
        yield status_html("Unexpected error – see the ffmpeg log tab.", kind="error"), \
            gr.update(), traceback.format_exc(), *nothing
    finally:
        # Also runs if the browser goes away and Gradio closes this generator.
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            if plan is not None:
                plan.partial_output.unlink(missing_ok=True)
        JOB.clear()
        RENDER_LOCK.release()


def render_preview(*vals):
    yield from _render(True, vals)


def render_full(*vals):
    yield from _render(False, vals)


def stop_render():
    proc = JOB.get("proc")
    if proc is not None and proc.poll() is None:
        JOB["cancel"] = True
        proc.terminate()
        return status_html("Stopping…", kind="error")
    return gr.update()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    hw = detect_hardware()
    on_drive = ON_COLAB
    default_clips = str(DRIVE_ROOT / "video" if on_drive else HERE / "video")
    default_mix = str(DRIVE_ROOT / "audio" if on_drive else HERE / "audio")
    default_out = str(DRIVE_ROOT / "output" if on_drive and Path("/content/drive").is_dir()
                      else HERE / "output")
    has_nvenc = hw["h264_nvenc"]

    with gr.Blocks(title="VisualSync") as demo:
        gr.Markdown("## VisualSync – beat-synced dance video")
        gr.Markdown(hardware_markdown(hw))

        with gr.Group():
            mode = gr.Radio([("Upload files", UPLOAD), ("Folders (Colab disk / Google Drive)", PATHS)],
                            value=UPLOAD, label="Sources")
            with gr.Column(visible=True) as upload_col:
                clip_files = gr.File(label="Dancer clips (same length, frame rate and size; "
                                           "played in name order)",
                                     file_count="multiple",
                                     file_types=sorted(mdv.VIDEO_EXTS))
                mix_file = gr.File(label="DJ Mix", file_count="single",
                                   file_types=sorted(mdv.AUDIO_EXTS))
            with gr.Column(visible=False) as paths_col:
                clips_dir = gr.Textbox(value=default_clips, label="Dancer clips folder")
                mix_path = gr.Textbox(value=default_mix,
                                      label="DJ Mix file (or a folder holding only the mix)")
            output_dir = gr.Textbox(
                value=default_out, label="Save videos in",
                info="Colab's disk is wiped when the session ends – a Google Drive folder "
                     "(/content/drive/MyDrive/…) keeps them.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Tempo")
                bpm = gr.Number(value=124, label="Mix tempo (BPM)", minimum=40, maximum=250)
                retempo = gr.Checkbox(value=False, label="Change the tempo")
                target_bpm = gr.Number(value=124, label="Output tempo (BPM)", minimum=40,
                                       maximum=250, interactive=False)
                stretch = gr.Radio(STRETCHES, value="keylock", label="Stretch", interactive=False)
                bars = gr.Number(value=0, precision=0, minimum=0, maximum=64,
                                 label="Bars per clip (0 = auto, closest to the clip's length)")
                beats_per_bar = gr.Number(value=4, precision=0, minimum=1, maximum=16,
                                          label="Beats per bar")
                offset = gr.Number(value=0.0, minimum=0, label="First downbeat in the mix (s)")
            with gr.Column():
                gr.Markdown("### Video & audio")
                resolution = gr.Dropdown(RESOLUTIONS, value="1920x1080", label="Resolution")
                # HEVC when the GPU encodes it; on the CPU, x265 is far slower than x264.
                default_codec = "hevc" if hw["hevc_nvenc"] else "h264"
                codec = gr.Dropdown(CODECS, value=default_codec, label="Codec")
                cq = gr.Slider(10, 40, value=DEFAULT_CQ[default_codec], step=1,
                               label="Quality (NVENC CQ / CPU CRF) – lower = better, bigger",
                               info="For YouTube: 24 with HEVC, 21 with H.264. Each step ≈ 10% size.")
                preset = gr.Dropdown(PRESETS, value="p7" if has_nvenc else "p4",
                                     label="Preset (on CPU: p1-p7 = x264 ultrafast-slow)")
                audio_bitrate = gr.Dropdown(AUDIO_BITRATES, value="320k", label="Audio (AAC)")
                fps = gr.Number(value=0, minimum=0, label="Frame rate (0 = same as the clips)")
                with gr.Row():
                    fade = gr.Number(value=4.0, minimum=0, label="Fade to black (s)")
                    audio_fade = gr.Checkbox(value=False, label="Fade audio too")
                with gr.Row():
                    encode = gr.Radio(HW_CHOICES, value="auto", label="Encode on")
                    decode = gr.Radio(HW_CHOICES, value="auto", label="Decode & scale on")

        with gr.Accordion("Logo overlay (centred over the intro)", open=False):
            overlay_on = gr.Checkbox(value=DEFAULT_LOGO.is_file(), label="Show a logo")
            with gr.Row():
                overlay = gr.Image(value=str(DEFAULT_LOGO) if DEFAULT_LOGO.is_file() else None,
                                   type="filepath", image_mode=None, format="png",
                                   label="Image (transparent PNG)", height=160)
                with gr.Column():
                    overlay_size = gr.Slider(5, 100, value=70, step=1, label="Width (% of video)")
                    overlay_start = gr.Number(value=1.0, minimum=0, label="Appears at (s)")
                    overlay_duration = gr.Number(value=5.0, minimum=0.5,
                                                 label="On screen for (s, including fades)")
                    with gr.Row():
                        overlay_fade_in = gr.Number(value=0.5, minimum=0, label="Fade in (s)")
                        overlay_fade_out = gr.Number(value=1.0, minimum=0, label="Fade out (s)")

        with gr.Row():
            preview_len = gr.Number(value=30, minimum=1, maximum=900, precision=0,
                                    label="Preview length (s)", scale=1)
            plan_btn = gr.Button("Check plan", scale=1)
            preview_btn = gr.Button("Render preview", variant="secondary", scale=1)
            render_btn = gr.Button("Render full video", variant="primary", scale=1)
            stop_btn = gr.Button("Stop", variant="stop", scale=1)

        status = gr.HTML(status_html("Ready."))
        with gr.Row():
            video_out = gr.Video(label="Preview", interactive=False)
            file_out = gr.File(label="Download", interactive=False)
        with gr.Tabs():
            with gr.Tab("Plan"):
                plan_box = gr.Textbox(lines=14, max_lines=30, show_label=False, interactive=False)
            with gr.Tab("ffmpeg log"):
                log_box = gr.Textbox(lines=14, max_lines=30, show_label=False, interactive=False)

        inputs = [mode, clip_files, mix_file, clips_dir, mix_path, output_dir,
                  bpm, retempo, target_bpm, stretch, bars, beats_per_bar, offset,
                  resolution, codec, cq, preset, audio_bitrate, fade, audio_fade, fps,
                  encode, decode,
                  overlay_on, overlay, overlay_size, overlay_start, overlay_duration,
                  overlay_fade_in, overlay_fade_out, preview_len]
        assert len(inputs) == len(FIELDS)
        outputs = [status, plan_box, log_box, video_out, file_out]

        mode.change(lambda m: (gr.update(visible=m == UPLOAD), gr.update(visible=m == PATHS)),
                    mode, [upload_col, paths_col])
        retempo.change(lambda on, b: (gr.update(interactive=True) if on
                                      else gr.update(interactive=False, value=b),
                                      gr.update(interactive=on)),
                       [retempo, bpm], [target_bpm, stretch])
        bpm.change(lambda b, on: gr.skip() if on else b, [bpm, retempo], target_bpm)
        # Switching codec moves the quality to that codec's default, unless it was customised.
        codec.change(lambda c, q: DEFAULT_CQ[c] if q in DEFAULT_CQ.values() else gr.skip(),
                     [codec, cq], cq)

        plan_btn.click(check_plan, inputs, [status, plan_box])
        preview_btn.click(render_preview, inputs, outputs)
        render_btn.click(render_full, inputs, outputs)
        stop_btn.click(stop_render, None, status)
    return demo


ALLOWED_ROOTS: list[Path] = []


def main() -> None:
    ap = argparse.ArgumentParser(description="VisualSync web UI")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--host", default=None, help="address to listen on (e.g. 0.0.0.0)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--allow-path", action="append", default=[], metavar="DIR",
                    help="extra folder the app may serve finished videos from (repeatable)")
    args = ap.parse_args()

    roots = [HERE, *map(Path, args.allow_path)]
    if ON_COLAB:
        roots.append(Path("/content"))
    ALLOWED_ROOTS[:] = [r.resolve() for r in roots]

    print("Checking ffmpeg and GPU …", flush=True)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(share=args.share, server_name=args.host, server_port=args.port,
                allowed_paths=[str(r) for r in ALLOWED_ROOTS], show_error=True)


if __name__ == "__main__":
    main()
