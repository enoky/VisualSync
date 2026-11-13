#!/usr/bin/env python3
"""
PySide6 front end for make_dance_video.py.

Probing, planning and the ffmpeg command all come from make_dance_video, so the
GUI and the command line always render identically. The GUI runs ffmpeg itself
(QProcess) so it can show progress and cancel the render instantly.

Launch with run_gui.bat, or:  .venv\\Scripts\\pythonw gui.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QSettings, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

import make_dance_video as mdv

HERE = mdv.HERE
APP_NAME = "Dance Video Sync"

RESOLUTIONS = [("720p  (1280×720)", "1280x720"), ("1080p  (1920×1080)", "1920x1080"),
               ("1440p  (2560×1440)", "2560x1440"), ("4K  (3840×2160)", "3840x2160")]
CODECS = [("H.264 – most compatible", "h264"), ("HEVC / H.265 – smaller files", "hevc")]
STRETCHES = [("Keylock – keep original pitch", "keylock"),
             ("Varispeed – pitch follows tempo", "varispeed")]
PRESETS = [(f"p{i}" + {7: " – best quality", 4: " – balanced", 1: " – fastest"}.get(i, ""), f"p{i}")
           for i in range(7, 0, -1)]
AUDIO_BITRATES = [("320 kbps", "320k"), ("256 kbps", "256k"), ("192 kbps", "192k")]


def hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def combo(items: list[tuple[str, object]]) -> QComboBox:
    box = QComboBox()
    for label, data in items:
        box.addItem(label, data)
    return box


def set_combo(box: QComboBox, data: object) -> None:
    i = box.findData(data)
    if i >= 0:
        box.setCurrentIndex(i)


def dspin(lo: float, hi: float, decimals: int, step: float, suffix: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setSuffix(suffix)
    sb.setKeyboardTracking(False)
    return sb


def row(*widgets, stretch: bool = True) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    if stretch:
        lay.addStretch(1)
    return w


class ProbeRelay(QObject):
    """Carries results from the probe worker thread back to the GUI thread."""
    finished = Signal(int, object, str)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.store = QSettings("AudioVisualSync", "DanceVideoSync")

        self.inputs: mdv.Inputs | None = None
        self.probe_error = ""
        self.probe_id = 0
        self.relay = ProbeRelay()
        self.relay.finished.connect(self.on_probe_done)

        self.proc: QProcess | None = None
        self.job_plan: mdv.Plan | None = None
        self.tmpdir: tempfile.TemporaryDirectory | None = None
        self.cancelled = False
        self.stdout_buf = b""
        self.stats: dict[str, str] = {}
        self.t_start = 0.0
        self.last_output: Path | None = None

        self.probe_timer = QTimer(self)
        self.probe_timer.setSingleShot(True)
        self.probe_timer.setInterval(400)
        self.probe_timer.timeout.connect(self.start_probe)
        self.plan_timer = QTimer(self)
        self.plan_timer.setSingleShot(True)
        self.plan_timer.setInterval(100)
        self.plan_timer.timeout.connect(self.update_plan)

        self.build_ui()
        self.load_settings()
        self.connect_signals()
        self.on_retempo_toggled(self.retempo_check.isChecked())
        self.start_probe()

    # ------------------------------------------------------------------ UI --
    def build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Sources ----------------------------------------------------------
        self.sources_box = QGroupBox("Sources")
        g = QGridLayout(self.sources_box)
        self.video_edit = QLineEdit()
        self.audio_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.output_edit.setClearButtonEnabled(True)
        self.output_edit.setToolTip("Leave empty to name the file automatically in ./output")
        video_btn = QPushButton("Browse…")
        audio_btn = QPushButton("Browse…")
        output_btn = QPushButton("Browse…")
        video_btn.clicked.connect(self.browse_video)
        audio_btn.clicked.connect(self.browse_audio)
        output_btn.clicked.connect(self.browse_output)
        for r, (label, edit, btn) in enumerate([("Dancer clips folder", self.video_edit, video_btn),
                                                ("DJ Mix", self.audio_edit, audio_btn),
                                                ("Save video as", self.output_edit, output_btn)]):
            g.addWidget(QLabel(label), r, 0)
            g.addWidget(edit, r, 1)
            g.addWidget(btn, r, 2)
        self.source_info = QLabel()
        self.source_info.setWordWrap(True)
        self.source_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        g.addWidget(self.source_info, 3, 1, 1, 2)
        root.addWidget(self.sources_box)

        # Tempo --------------------------------------------------------------
        mid = QHBoxLayout()
        self.tempo_box = QGroupBox("Tempo")
        f = QFormLayout(self.tempo_box)
        self.bpm_spin = dspin(40, 250, 2, 0.5, " BPM")
        self.bpm_spin.setToolTip("Tempo of the DJ Mix as recorded")
        self.retempo_check = QCheckBox("Change to")
        self.target_spin = dspin(40, 250, 2, 0.5, " BPM")
        self.target_spin.setToolTip("Tempo of the finished video. The mix is time-stretched to it.")
        self.stretch_combo = combo(STRETCHES)
        self.stretch_combo.setToolTip("Keylock keeps the original key (rubberband time-stretch).\n"
                                      "Varispeed changes speed and pitch together, like a turntable.")
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(0, 64)
        self.bars_spin.setSpecialValueText("Auto")
        self.bars_spin.setSuffix(" bars")
        self.bars_spin.setToolTip("Bars each clip is stretched to. Auto = closest to the clip's length.")
        self.offset_spin = dspin(0, 600, 3, 0.01, " s")
        self.offset_spin.setToolTip("Time of the first downbeat in the original mix")
        self.clip_bpm_spin = dspin(0, 250, 2, 0.5, " BPM")
        self.clip_bpm_spin.setSpecialValueText("Off")
        self.clip_bpm_spin.setToolTip("Tempo the dancers move to. Each clip is cut to exactly whole bars\n"
                                      "of it (e.g. 8 bars = 15.000 s at 128 BPM), so the moves land on the beat.")
        self.trim_spin = QSpinBox()
        self.trim_spin.setRange(0, 60)
        self.trim_spin.setSpecialValueText("Off")
        self.trim_spin.setSuffix(" frames")
        self.trim_spin.setToolTip("Drop the last frames of every clip (e.g. glitchy AI-generated endings).\n"
                                  "Counted at the clips' own frame rate: with 48 fps clips, "
                                  "2 frames = 1 frame of a 24 fps video.")
        f.addRow("Mix tempo", self.bpm_spin)
        f.addRow("Output tempo", row(self.retempo_check, self.target_spin))
        f.addRow("Stretch", self.stretch_combo)
        f.addRow("Clip length", self.bars_spin)
        f.addRow("First downbeat", self.offset_spin)
        f.addRow("Dancer tempo", self.clip_bpm_spin)
        f.addRow("Trim clip ends", self.trim_spin)
        mid.addWidget(self.tempo_box, 1)

        # Encoding -----------------------------------------------------------
        self.encode_box = QGroupBox("Video && audio")
        f2 = QFormLayout(self.encode_box)
        self.res_combo = combo(RESOLUTIONS)
        self.codec_combo = combo(CODECS)
        self.cq_spin = QSpinBox()
        self.cq_spin.setRange(10, 40)
        self.cq_spin.setToolTip("NVENC constant quality. Lower = better quality and bigger file.\n"
                                "For YouTube uploads: about 24 with HEVC, 21 with H.264.")
        self.preset_combo = combo(PRESETS)
        self.abr_combo = combo(AUDIO_BITRATES)
        self.fade_spin = dspin(0, 30, 1, 0.5, " s")
        self.audio_fade_check = QCheckBox("fade audio too")
        self.cpu_check = QCheckBox("Decode && scale on CPU (if GPU decoding gives trouble)")
        f2.addRow("Resolution", self.res_combo)
        f2.addRow("Codec", self.codec_combo)
        f2.addRow("Quality (CQ)", row(self.cq_spin, QLabel("lower = better")))
        f2.addRow("NVENC preset", self.preset_combo)
        f2.addRow("Audio (AAC)", self.abr_combo)
        f2.addRow("Fade to black", row(self.fade_spin, self.audio_fade_check))
        f2.addRow("", self.cpu_check)
        mid.addWidget(self.encode_box, 1)
        root.addLayout(mid)

        # Logo overlay ---------------------------------------------------------
        self.overlay_box = QGroupBox("Logo overlay (centred over the intro)")
        self.overlay_box.setCheckable(True)
        og = QGridLayout(self.overlay_box)
        self.overlay_edit = QLineEdit()
        self.overlay_edit.setToolTip("A transparent PNG, e.g. subscribe.png")
        overlay_btn = QPushButton("Browse…")
        overlay_btn.clicked.connect(self.browse_overlay)
        self.overlay_thumb = QLabel()
        self.overlay_thumb.setFixedSize(144, 72)
        self.overlay_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_thumb.setStyleSheet("background: #1e1e1e; border-radius: 4px; color: #888;")
        self.overlay_size_spin = dspin(5, 100, 0, 5, " % of width")
        self.overlay_size_spin.setToolTip("Image width as a percentage of the video width")
        self.overlay_start_spin = dspin(0, 600, 2, 0.5, " s")
        self.overlay_dur_spin = dspin(0.5, 600, 2, 0.5, " s")
        self.overlay_dur_spin.setToolTip("Total time on screen, including the fades")
        self.overlay_fin_spin = dspin(0, 10, 2, 0.25, " s")
        self.overlay_fout_spin = dspin(0, 10, 2, 0.25, " s")
        og.addWidget(QLabel("Image"), 0, 0)
        og.addWidget(self.overlay_edit, 0, 1)
        og.addWidget(overlay_btn, 0, 2)
        og.addWidget(QLabel("Size"), 1, 0)
        og.addWidget(row(self.overlay_size_spin, QLabel("  appears at"), self.overlay_start_spin,
                         QLabel("  on screen for"), self.overlay_dur_spin,
                         QLabel("  fade in"), self.overlay_fin_spin,
                         QLabel("  fade out"), self.overlay_fout_spin), 1, 1, 1, 2)
        og.addWidget(self.overlay_thumb, 0, 3, 2, 1)
        og.setColumnStretch(1, 1)
        root.addWidget(self.overlay_box)

        # Plan / log ----------------------------------------------------------
        mono = QFont("Consolas", 9)
        mono.setStyleHint(QFont.StyleHint.TypeWriter)   # falls back to any monospace font
        self.tabs = QTabWidget()
        self.plan_view = QPlainTextEdit(readOnly=True)
        self.log_view = QPlainTextEdit(readOnly=True)
        for view in (self.plan_view, self.log_view):
            view.setFont(mono)
            view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tabs.addTab(self.plan_view, "Plan")
        self.tabs.addTab(self.log_view, "ffmpeg log")
        root.addWidget(self.tabs, 1)

        # Progress -----------------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.status_label = QLabel("Ready")
        root.addWidget(self.progress)
        root.addWidget(self.status_label)

        # Buttons ------------------------------------------------------------
        self.preview_spin = QSpinBox()
        self.preview_spin.setRange(5, 900)
        self.preview_spin.setSuffix(" s")
        self.preview_btn = QPushButton("Render preview")
        self.preview_btn.setToolTip("Render only the start, to check sync and quality quickly")
        self.render_btn = QPushButton("Render full video")
        self.render_btn.setDefault(True)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.play_btn = QPushButton("Play result")
        self.play_btn.setEnabled(False)
        self.folder_btn = QPushButton("Open output folder")
        self.preview_btn.clicked.connect(lambda: self.start_render(preview=True))
        self.render_btn.clicked.connect(lambda: self.start_render(preview=False))
        self.cancel_btn.clicked.connect(self.cancel_render)
        self.play_btn.clicked.connect(self.play_result)
        self.folder_btn.clicked.connect(self.open_folder)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Preview length"))
        bar.addWidget(self.preview_spin)
        bar.addWidget(self.preview_btn)
        bar.addSpacing(12)
        bar.addWidget(self.render_btn)
        bar.addWidget(self.cancel_btn)
        bar.addStretch(1)
        bar.addWidget(self.play_btn)
        bar.addWidget(self.folder_btn)
        root.addLayout(bar)

        self.settings_widgets = [self.sources_box, self.tempo_box, self.encode_box, self.overlay_box,
                                 self.preview_spin, self.preview_btn, self.render_btn]
        self.resize(1000, 860)

    def connect_signals(self) -> None:
        self.video_edit.textChanged.connect(self.probe_timer.start)
        self.audio_edit.textChanged.connect(self.probe_timer.start)
        self.output_edit.textChanged.connect(self.plan_timer.start)
        for sb in (self.bpm_spin, self.target_spin, self.bars_spin, self.offset_spin, self.trim_spin,
                   self.clip_bpm_spin,
                   self.cq_spin, self.fade_spin, self.preview_spin, self.overlay_size_spin,
                   self.overlay_start_spin, self.overlay_dur_spin, self.overlay_fin_spin,
                   self.overlay_fout_spin):
            sb.valueChanged.connect(self.plan_timer.start)
        self.overlay_box.toggled.connect(self.plan_timer.start)
        self.overlay_edit.textChanged.connect(self.plan_timer.start)
        self.overlay_edit.textChanged.connect(self.update_overlay_thumb)
        for cb in (self.stretch_combo, self.res_combo, self.codec_combo, self.preset_combo,
                   self.abr_combo):
            cb.currentIndexChanged.connect(self.plan_timer.start)
        for ch in (self.retempo_check, self.audio_fade_check, self.cpu_check):
            ch.toggled.connect(self.plan_timer.start)
        self.retempo_check.toggled.connect(self.on_retempo_toggled)
        self.bpm_spin.valueChanged.connect(self.on_bpm_changed)

    def on_retempo_toggled(self, on: bool) -> None:
        self.target_spin.setEnabled(on)
        self.stretch_combo.setEnabled(on)
        if not on:
            self.target_spin.setValue(self.bpm_spin.value())

    def on_bpm_changed(self, value: float) -> None:
        if not self.retempo_check.isChecked():
            self.target_spin.setValue(value)

    # ------------------------------------------------------------ settings --
    def load_settings(self) -> None:
        st = self.store
        default_audio = ""
        try:
            default_audio = str(mdv.find_audio(HERE / "audio"))
        except mdv.RenderError:
            pass
        self.video_edit.setText(st.value("video_dir", str(HERE / "video")))
        self.audio_edit.setText(st.value("audio", default_audio))
        self.output_edit.setText(st.value("output", ""))
        self.bpm_spin.setValue(float(st.value("bpm", 124.0)))
        self.retempo_check.setChecked(st.value("retempo", False, type=bool))
        self.target_spin.setValue(float(st.value("target_bpm", self.bpm_spin.value())))
        set_combo(self.stretch_combo, st.value("stretch", "keylock"))
        self.bars_spin.setValue(int(st.value("bars", 0)))
        self.offset_spin.setValue(float(st.value("offset", 0.0)))
        self.trim_spin.setValue(int(st.value("trim_end", 0)))
        self.clip_bpm_spin.setValue(float(st.value("clip_bpm", 128.0)))
        set_combo(self.res_combo, "1920x1080")
        set_combo(self.res_combo, st.value("resolution", "1920x1080"))
        set_combo(self.codec_combo, st.value("codec", "hevc"))
        self.cq_spin.setValue(int(st.value("cq", 24)))
        set_combo(self.preset_combo, st.value("preset", "p7"))
        set_combo(self.abr_combo, st.value("audio_bitrate", "320k"))
        self.fade_spin.setValue(float(st.value("fade", 4.0)))
        self.audio_fade_check.setChecked(st.value("audio_fade", False, type=bool))
        self.cpu_check.setChecked(st.value("cpu_decode", False, type=bool))
        self.preview_spin.setValue(int(st.value("preview", 30)))
        default_logo = HERE / "subscribe.png"
        self.overlay_box.setChecked(st.value("overlay_enabled", default_logo.is_file(), type=bool))
        self.overlay_edit.setText(st.value("overlay", str(default_logo) if default_logo.is_file() else ""))
        self.overlay_size_spin.setValue(float(st.value("overlay_size", 70.0)))
        self.overlay_start_spin.setValue(float(st.value("overlay_start", 1.0)))
        self.overlay_dur_spin.setValue(float(st.value("overlay_duration", 5.0)))
        self.overlay_fin_spin.setValue(float(st.value("overlay_fade_in", 0.5)))
        self.overlay_fout_spin.setValue(float(st.value("overlay_fade_out", 1.0)))
        self.update_overlay_thumb()
        geom = st.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)

    def save_settings(self) -> None:
        st = self.store
        for key, value in {
            "video_dir": self.video_edit.text(), "audio": self.audio_edit.text(),
            "output": self.output_edit.text(), "bpm": self.bpm_spin.value(),
            "retempo": self.retempo_check.isChecked(), "target_bpm": self.target_spin.value(),
            "stretch": self.stretch_combo.currentData(), "bars": self.bars_spin.value(),
            "offset": self.offset_spin.value(), "trim_end": self.trim_spin.value(),
            "clip_bpm": self.clip_bpm_spin.value(),
            "resolution": self.res_combo.currentData(),
            "codec": self.codec_combo.currentData(), "cq": self.cq_spin.value(),
            "preset": self.preset_combo.currentData(),
            "audio_bitrate": self.abr_combo.currentData(), "fade": self.fade_spin.value(),
            "audio_fade": self.audio_fade_check.isChecked(),
            "cpu_decode": self.cpu_check.isChecked(), "preview": self.preview_spin.value(),
            "overlay_enabled": self.overlay_box.isChecked(), "overlay": self.overlay_edit.text(),
            "overlay_size": self.overlay_size_spin.value(),
            "overlay_start": self.overlay_start_spin.value(),
            "overlay_duration": self.overlay_dur_spin.value(),
            "overlay_fade_in": self.overlay_fin_spin.value(),
            "overlay_fade_out": self.overlay_fout_spin.value(),
            "geometry": self.saveGeometry(),
        }.items():
            st.setValue(key, value)
        st.sync()

    def current_settings(self, preview: float | None) -> mdv.Settings:
        w, h = (int(v) for v in self.res_combo.currentData().split("x"))
        out = self.output_edit.text().strip()
        overlay = None
        if self.overlay_box.isChecked():
            text = self.overlay_edit.text().strip()
            if not text:
                raise mdv.RenderError("Choose a logo image, or untick “Logo overlay”.")
            overlay = Path(text)
        return mdv.Settings(
            overlay=overlay,
            overlay_size=self.overlay_size_spin.value(),
            overlay_start=self.overlay_start_spin.value(),
            overlay_duration=self.overlay_dur_spin.value(),
            overlay_fade_in=self.overlay_fin_spin.value(),
            overlay_fade_out=self.overlay_fout_spin.value(),
            bpm=self.bpm_spin.value(),
            target_bpm=self.target_spin.value() if self.retempo_check.isChecked() else None,
            stretch=self.stretch_combo.currentData(),
            bars=self.bars_spin.value() or None,
            offset=self.offset_spin.value(),
            trim_end=self.trim_spin.value(),
            clip_bpm=self.clip_bpm_spin.value() or None,
            width=w, height=h,
            fade=self.fade_spin.value(),
            audio_fade=self.audio_fade_check.isChecked(),
            codec=self.codec_combo.currentData(),
            cq=self.cq_spin.value(),
            preset=self.preset_combo.currentData(),
            audio_bitrate=self.abr_combo.currentData(),
            cpu_decode=self.cpu_check.isChecked(),
            preview=preview,
            output=Path(out) if out else None,
        )

    # -------------------------------------------------------------- browse --
    def browse_video(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Dancer clips folder", self.video_edit.text())
        if d:
            self.video_edit.setText(d)

    def browse_audio(self) -> None:
        start = self.audio_edit.text() or str(HERE / "audio")
        exts = " ".join(f"*{e}" for e in sorted(mdv.AUDIO_EXTS))
        f, _ = QFileDialog.getOpenFileName(self, "DJ Mix", start, f"Audio ({exts});;All files (*)")
        if f:
            self.audio_edit.setText(f)

    def browse_overlay(self) -> None:
        start = self.overlay_edit.text() or str(HERE)
        f, _ = QFileDialog.getOpenFileName(self, "Logo image", start,
                                           "Images (*.png *.webp *.tif *.tiff);;All files (*)")
        if f:
            self.overlay_edit.setText(f)
            self.overlay_box.setChecked(True)

    def update_overlay_thumb(self) -> None:
        pix = QPixmap(self.overlay_edit.text().strip())
        if pix.isNull():
            self.overlay_thumb.setPixmap(QPixmap())
            self.overlay_thumb.setText("no image")
            return
        size = self.overlay_thumb.size()
        self.overlay_thumb.setPixmap(pix.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))

    def browse_output(self) -> None:
        start = self.output_edit.text() or self.output_edit.placeholderText()
        f, _ = QFileDialog.getSaveFileName(self, "Save video as", start, "MP4 video (*.mp4)",
                                           options=QFileDialog.Option.DontConfirmOverwrite)
        if f:
            self.output_edit.setText(f)

    # --------------------------------------------------------------- probe --
    def start_probe(self) -> None:
        self.probe_timer.stop()
        self.probe_id += 1
        pid = self.probe_id
        video = Path(self.video_edit.text().strip() or ".")
        audio_text = self.audio_edit.text().strip()
        self.inputs = None
        self.probe_error = ""
        self.source_info.setStyleSheet("")
        self.source_info.setText("Reading clips and mix…")
        self.update_plan()
        if not audio_text:
            self.on_probe_done(pid, None, "Choose the DJ Mix file.")
            return
        audio = Path(audio_text)
        relay = self.relay

        def job() -> None:
            try:
                relay.finished.emit(pid, mdv.probe_inputs(video, audio), "")
            except mdv.RenderError as exc:
                relay.finished.emit(pid, None, str(exc))
            except Exception as exc:  # noqa: BLE001 - shown to the user
                relay.finished.emit(pid, None, f"{type(exc).__name__}: {exc}")

        QThreadPool.globalInstance().start(job)

    def on_probe_done(self, pid: int, inputs: object, error: str) -> None:
        if pid != self.probe_id:
            return  # a newer probe superseded this one
        self.inputs = inputs if isinstance(inputs, mdv.Inputs) else None
        self.probe_error = error
        if self.inputs:
            self.source_info.setStyleSheet("")
            self.source_info.setText(self.inputs.describe())
        else:
            self.source_info.setStyleSheet("color: #d9534f;")
            self.source_info.setText(error)
        self.update_plan()

    # ---------------------------------------------------------------- plan --
    def update_plan(self) -> None:
        self.plan_timer.stop()
        if self.proc is not None:
            return
        ok = False
        if self.inputs is None:
            self.plan_view.setPlainText(self.probe_error or "Waiting for clips and mix…")
            self.output_edit.setPlaceholderText("")
        else:
            try:
                plan = mdv.make_plan(self.inputs, self.current_settings(None))
            except mdv.RenderError as exc:
                self.plan_view.setPlainText(f"Can't render with these settings:\n\n{exc}")
            else:
                ok = True
                self.output_edit.setPlaceholderText(str(plan.output))
                n = self.preview_spin.value()
                lines = [*plan.summary(), "",
                         f"  Preview     : first {n} s -> "
                         f"{plan.output.stem}_preview{min(n, int(plan.target_dur))}s.mp4"]
                self.plan_view.setPlainText("\n".join(line[2:] for line in lines))
        self.preview_btn.setEnabled(ok)
        self.render_btn.setEnabled(ok)

    # -------------------------------------------------------------- render --
    def start_render(self, preview: bool) -> None:
        if self.inputs is None or self.proc is not None:
            return
        try:
            plan = mdv.make_plan(self.inputs,
                                 self.current_settings(self.preview_spin.value() if preview else None))
        except mdv.RenderError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return

        if plan.output.exists():
            answer = QMessageBox.question(
                self, APP_NAME, f"{plan.output.name} already exists.\n\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            plan.output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Can't create the output folder:\n{exc}")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            note = mdv.check_plan_encoder(plan)
        except mdv.RenderError as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.tmpdir = tempfile.TemporaryDirectory(prefix="dancesync_")
        list_file = Path(self.tmpdir.name) / "clips.txt"
        list_file.write_text(plan.concat_text(), encoding="utf-8")
        cmd = plan.command(list_file, plan.partial_output)
        cmd[0] = shutil.which("ffmpeg") or "ffmpeg"

        self.log_view.setPlainText((note + "\n\n" if note else "") + subprocess.list2cmdline(cmd) + "\n")
        self.job_plan = plan
        self.cancelled = False
        self.stdout_buf = b""
        self.stats = {}
        self.progress.setValue(0)

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(HERE))
        proc.readyReadStandardOutput.connect(self.on_stdout)
        proc.readyReadStandardError.connect(self.on_stderr)
        proc.finished.connect(self.on_finished)
        proc.errorOccurred.connect(self.on_proc_error)
        self.proc = proc
        self.set_busy(True)
        kind = f"preview ({plan.target_dur:g} s)" if preview else "full video"
        self.status_label.setText(f"Starting {kind} → {plan.output.name}")
        self.t_start = time.monotonic()
        proc.start()

    def set_busy(self, busy: bool) -> None:
        for w in self.settings_widgets:
            w.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        if not busy:
            self.on_retempo_toggled(self.retempo_check.isChecked())

    def on_stdout(self) -> None:
        if self.proc is None:
            return
        self.stdout_buf += bytes(self.proc.readAllStandardOutput())
        *lines, self.stdout_buf = self.stdout_buf.split(b"\n")
        for raw in lines:
            key, _, val = raw.decode("utf-8", "replace").strip().partition("=")
            self.stats[key] = val
            if key == "progress":
                self.show_progress()

    def show_progress(self) -> None:
        plan = self.job_plan
        if plan is None:
            return
        us = self.stats.get("out_time_us", "")
        done = min(plan.target_dur, max(0, int(us)) / 1e6) if us.lstrip("-").isdigit() else 0.0
        frac = done / plan.target_dur if plan.target_dur else 0.0
        self.progress.setValue(int(frac * 1000))
        elapsed = time.monotonic() - self.t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (plan.target_dur - done) / rate if rate > 0 else 0.0
        fps = self.stats.get("fps", "?")
        self.status_label.setText(
            f"{hms(done)} / {hms(plan.target_dur)}   ·   {fps} fps  ({rate:.1f}× real time)   ·   "
            f"elapsed {hms(elapsed)}   ·   about {hms(eta)} left")

    def on_stderr(self) -> None:
        if self.proc is None:
            return
        text = bytes(self.proc.readAllStandardError()).decode("utf-8", "replace")
        if text.strip():
            self.log_view.appendPlainText(text.rstrip())

    def on_proc_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.log_view.appendPlainText(f"Could not start ffmpeg: {self.proc.errorString()}")
            self.finish(False, "Could not start ffmpeg – see the log")

    def on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self.proc is None:
            return
        self.on_stderr()
        elapsed = hms(time.monotonic() - self.t_start)
        if self.cancelled:
            self.finish(False, "Cancelled")
        elif status == QProcess.ExitStatus.NormalExit and code == 0:
            self.finish(True, f"Finished in {elapsed} → {self.job_plan.output}")
        else:
            self.finish(False, f"ffmpeg failed (exit code {code}) – see the ffmpeg log")
            self.tabs.setCurrentWidget(self.log_view)

    def finish(self, success: bool, message: str) -> None:
        plan, proc = self.job_plan, self.proc
        self.proc = None
        self.job_plan = None
        if proc is not None:
            proc.deleteLater()
        if plan is not None:
            try:
                mdv.finish_output(plan, success)
            except OSError as exc:
                success = False
                message = (f"Render finished, but {plan.partial_output.name} could not be renamed "
                           f"({exc}). Is {plan.output.name} open in a player?")
        if self.tmpdir is not None:
            self.tmpdir.cleanup()
            self.tmpdir = None
        if success and plan is not None:
            self.last_output = plan.output
            self.progress.setValue(1000)
            self.play_btn.setEnabled(True)
        elif not success:
            self.progress.setValue(0)
        self.status_label.setText(message)
        self.set_busy(False)
        self.update_plan()

    def cancel_render(self) -> None:
        if self.proc is not None:
            self.cancelled = True
            self.status_label.setText("Cancelling…")
            self.proc.kill()

    def play_result(self) -> None:
        if self.last_output and self.last_output.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output)))

    def open_folder(self) -> None:
        out = self.output_edit.text().strip() or self.output_edit.placeholderText()
        folder = Path(out).parent if out else HERE / "output"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
        if self.proc is not None:
            answer = QMessageBox.question(
                self, APP_NAME, "A render is still running.\n\nCancel it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancelled = True
            proc = self.proc
            proc.kill()
            proc.waitForFinished(5000)
            if self.proc is not None:   # finished signal didn't arrive in time
                self.finish(False, "Cancelled")
        self.save_settings()
        event.accept()


def install_excepthook() -> None:
    """With pythonw there's no console, so show unexpected errors in a dialog."""
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.__stderr__ and sys.__stderr__.write(text)
        QMessageBox.critical(None, APP_NAME, f"Unexpected error:\n\n{text}")
    sys.excepthook = hook


def main() -> int:
    if os.name == "nt":  # own taskbar entry/icon instead of grouping under python.exe
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AudioVisualSync.DanceVideoSync")
        except (AttributeError, OSError):
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    install_excepthook()
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
