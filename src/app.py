"""A single-window humming recorder and instrument player.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPen
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QToolButton, QVBoxLayout, QWidget,
)

from engine import DRUMS, INSTRUMENTS, MAX_SECONDS, Note, SAMPLE_RATE, export_midi, read_audio, write_json
from game_library import SoundLibrary, load_library_path

APP = Path(__file__).resolve().parent
# Keep the existing storage and instance identifiers so upgrades retain saved takes.
DATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LocalAudioTools" / "Hum to Melody"
ROOT = DATA
CACHE = ROOT / "cache"
STATE = ROOT / "last-session.json"
PREFERENCES = ROOT / "preferences.json"
for folder in (CACHE, ROOT / "Recordings", ROOT / "Sessions", ROOT / "Exports", ROOT / "Logs"):
    folder.mkdir(parents=True, exist_ok=True)
LOG = logging.getLogger("hum-to-music")
LOG.setLevel(logging.INFO)
handler = RotatingFileHandler(ROOT / "Logs" / "custom-app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
LOG.addHandler(handler)
library_config = ROOT / "library.json"
SOUND_LIBRARY = SoundLibrary(load_library_path(library_config) if library_config.is_file()
                             else Path.home() / "Music" / "Hum to Melody")


class AudioRecorder:
    """The microphone is opened only when the Record button is pressed."""
    def __init__(self):
        self.stream = None
        self.frames = []
        self.peak = 0.
        self.rate = 48000
        self.started = 0.
        self.overflows = 0

    def start(self, device=None):
        if self.stream:
            raise RuntimeError("Already recording")
        information = sd.query_devices(device, "input")
        self.rate = round(information["default_samplerate"])
        self.frames, self.peak, self.overflows = [], 0., 0

        def callback(data, _frames, _time, status):
            if status.input_overflow:
                self.overflows += 1
            self.frames.append(data[:, 0].copy())
            self.peak = float(np.max(np.abs(data)))

        self.stream = sd.InputStream(device=device, samplerate=self.rate, channels=1,
                                     dtype="float32", callback=callback, latency="high")
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            self.stream = None
            raise
        self.started = time.monotonic()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        frames, self.frames = self.frames, []
        return (np.concatenate(frames) if frames else np.empty(0, dtype=np.float32)), self.rate


class AudioJob(QThread):
    completed = Signal(dict)

    def __init__(self, job, context, parent=None):
        super().__init__(parent)
        self.job, self.context = job, context
        self.process = None
        self.cancelled = False

    def run(self):
        job_file = CACHE / f"job-{uuid.uuid4().hex}.json"
        write_json(job_file, self.job)
        try:
            interpreter = str(Path(sys.executable).with_name("python.exe"))
            self.process = subprocess.Popen([interpreter, str(APP / "engine.py"), "--job", str(job_file)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, encoding="utf-8", errors="replace",
                                            creationflags=subprocess.CREATE_NO_WINDOW, cwd=str(APP))
            stdout, stderr = self.process.communicate(timeout=120)
            if self.cancelled:
                return
            result_lines = [line[10:] for line in stdout.splitlines() if line.startswith("HM_RESULT:")]
            if not result_lines:
                LOG.error("Worker failed (%s): %s %s", self.process.returncode, stderr[-5000:], stdout[-2000:])
                result = {"ok": False, "error": "The sound engine stopped unexpectedly. Please try again."}
            else:
                result = json.loads(result_lines[-1])
                if not result["ok"]:
                    LOG.error("Audio job failed: %s", stderr[-5000:])
            result["context"] = self.context
            self.completed.emit(result)
        except Exception:
            LOG.exception("Audio worker could not complete")
            if self.process and self.process.poll() is None:
                self.process.kill()
                self.process.communicate()
            if not self.cancelled:
                self.completed.emit({"ok": False, "error": "I couldn't finish that sound. Please try again.", "context": self.context})
        finally:
            job_file.unlink(missing_ok=True)

    def cancel(self):
        self.cancelled = True
        if self.process and self.process.poll() is None:
            self.process.kill()


class Waveform(QWidget):
    def __init__(self):
        super().__init__()
        self.peaks = np.zeros(120)
        self.has_audio = False
        self.live = False
        self.setMinimumHeight(100)
        self.setAccessibleName("Recording waveform")

    def set_audio(self, audio):
        self.has_audio = len(audio) > 0
        bins = np.array_split(np.asarray(audio), 120) if self.has_audio else []
        self.peaks = np.array([np.max(np.abs(b)) if len(b) else 0 for b in bins]) if bins else np.zeros(120)
        maximum = max(float(np.max(self.peaks)), .01)
        self.peaks = np.minimum(1., self.peaks / maximum)
        self.live = False
        self.update()

    def live_level(self, level):
        self.live = True
        self.peaks = np.roll(self.peaks, -1)
        self.peaks[-1] = min(1., level * 3)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f4f8f7"))
        if not self.has_audio and not self.live:
            painter.setPen(QColor("#7d8b8b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Your melody will appear here")
            return
        width = self.width() - 32
        center = self.height() / 2
        painter.setPen(QPen(QColor("#d35f50" if self.live else "#25867d"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for index, peak in enumerate(self.peaks):
            x = round(16 + (index + .5) * width / len(self.peaks))
            height = max(2., float(peak) ** .6 * (self.height() - 22) / 2)
            painter.drawLine(x, round(center - height), x, round(center + height))


class HumToMusic(QMainWindow):
    def __init__(self, restore=True):
        super().__init__()
        self.setWindowTitle("Hum to Tune")
        self.setWindowIcon(QIcon(str(APP / "icon.svg")))
        self.resize(790, 875)
        self.setMinimumSize(670, 650)
        self.recorder = AudioRecorder()
        self.notes = []
        self.analysis_mode = "melody"
        self.melody_patch = 0
        self.drum_note = 38
        self.source = None
        self.notes_file = None
        self.rendered = None
        self.job = None
        self.jobs = []
        self.revision = 0
        self.playing = None
        self.play_buffer = None
        self.play_source = None
        self.pending_autoplay = False
        self.closing = False
        self._build_ui()
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(350)
        self.debounce.timeout.connect(lambda: self._start_render(True))
        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._load_preferences()
        self._refresh_devices()
        self._connect_controls()
        self._update_buttons()
        if restore:
            QTimer.singleShot(100, self._restore_session)
        # Original recordings and saved notes are never treated as a cache.
        cached_audio = sorted(CACHE.glob("sound-*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_audio in cached_audio[8:]:
            try:
                old_audio.unlink()
            except OSError:
                pass

    def _build_ui(self):
        shell = QWidget()
        shell.setObjectName("shell")
        outer = QVBoxLayout(shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(shell)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)
        page = QWidget()
        scroll.setWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(34, 28, 34, 22)
        layout.setSpacing(18)
        title = QLabel("Hum to Tune")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Hum a tune or tap a rhythm. Choose a sound. Hear it back.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        record_card, record_layout = self._card()
        layout.addWidget(record_card)
        record_layout.addWidget(self._heading("1   Record your sound"))
        hint = QLabel("Humming becomes notes. Tapping becomes drum hits.")
        hint.setObjectName("muted")
        record_layout.addWidget(hint)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Recording type"))
        self.input_mode = QComboBox()
        self.input_mode.setAccessibleName("Recording type")
        for name, mode in [("Automatic — humming or taps", "auto"),
                           ("Humming / melody", "melody"), ("Taps / percussion", "percussion")]:
            self.input_mode.addItem(name, mode)
        mode_row.addWidget(self.input_mode, 1)
        record_layout.addLayout(mode_row)
        row = QHBoxLayout()
        self.record_button = QPushButton("●   Record humming")
        self.record_button.setObjectName("record")
        self.record_button.setAccessibleName("Record humming")
        self.record_button.setMinimumHeight(52)
        self.record_button.setMinimumWidth(235)
        row.addWidget(self.record_button)
        self.duration_label = QLabel("Ready when you are")
        self.duration_label.setObjectName("muted")
        row.addWidget(self.duration_label, 1, Qt.AlignmentFlag.AlignRight)
        record_layout.addLayout(row)
        self.waveform = Waveform()
        record_layout.addWidget(self.waveform)
        source_row = QHBoxLayout()
        self.original_button = QPushButton("Hear original")
        self.original_button.setObjectName("secondary")
        self.original_button.setAccessibleName("Hear original recording")
        self.original_button.setEnabled(False)
        source_row.addWidget(self.original_button)
        source_row.addStretch()
        self.import_button = QPushButton("Open a recording…")
        self.import_button.setObjectName("link")
        source_row.addWidget(self.import_button)
        self.example_button = QPushButton("Try an example")
        self.example_button.setObjectName("link")
        source_row.addWidget(self.example_button)
        record_layout.addLayout(source_row)

        sound_card, sound_layout = self._card()
        layout.addWidget(sound_card)
        sound_layout.addWidget(self._heading("2   Choose a sound"))
        sound_row = QHBoxLayout()
        self.instrument = QComboBox()
        self.instrument.setAccessibleName("Instrument")
        self.instrument.setMinimumHeight(45)
        for name, patch in INSTRUMENTS:
            self.instrument.addItem(name, patch)
        sound_row.addWidget(self.instrument, 1)
        self.listen_button = QPushButton("▶   Listen")
        self.listen_button.setObjectName("listen")
        self.listen_button.setAccessibleName("Listen to instrument")
        self.listen_button.setMinimumSize(145, 45)
        sound_row.addWidget(self.listen_button)
        sound_layout.addLayout(sound_row)
        self.sound_hint = QLabel("Your instrument plays automatically after you finish recording.")
        self.sound_hint.setObjectName("muted")
        self.sound_hint.setWordWrap(True)
        sound_layout.addWidget(self.sound_hint)
        volume_row = QHBoxLayout()
        volume_row.addWidget(QLabel("Volume"))
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setAccessibleName("Playback volume")
        self.volume.setRange(0, 100)
        self.volume.setValue(75)
        self.volume.setMaximumWidth(190)
        volume_row.addWidget(self.volume)
        volume_row.addStretch()
        self.adjust_button = QToolButton()
        self.adjust_button.setText("Adjust sound")
        self.adjust_button.setCheckable(True)
        self.adjust_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adjust_button.setArrowType(Qt.ArrowType.RightArrow)
        volume_row.addWidget(self.adjust_button)
        sound_layout.addLayout(volume_row)
        self.adjustments = QWidget()
        adjust_layout = QVBoxLayout(self.adjustments)
        adjust_layout.setContentsMargins(0, 8, 0, 0)
        speed_row = QHBoxLayout()
        self.speed_label = QLabel("Speed 100%")
        self.speed_label.setMinimumWidth(115)
        speed_row.addWidget(self.speed_label)
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setAccessibleName("Melody speed")
        self.speed.setRange(50, 150)
        self.speed.setValue(100)
        speed_row.addWidget(self.speed)
        adjust_layout.addLayout(speed_row)
        pitch_row = QHBoxLayout()
        label = QLabel("Pitch")
        label.setMinimumWidth(115)
        pitch_row.addWidget(label)
        self.octave = QComboBox()
        self.octave.setAccessibleName("Octave")
        for text, value in [("Two octaves lower", -2), ("One octave lower", -1), ("Original pitch", 0),
                            ("One octave higher", 1), ("Two octaves higher", 2)]:
            self.octave.addItem(text, value)
        self.octave.setCurrentIndex(2)
        pitch_row.addWidget(self.octave, 1)
        adjust_layout.addLayout(pitch_row)
        room_row = QHBoxLayout()
        self.room_label = QLabel("Room 15%")
        self.room_label.setMinimumWidth(115)
        room_row.addWidget(self.room_label)
        self.room = QSlider(Qt.Orientation.Horizontal)
        self.room.setAccessibleName("Room sound")
        self.room.setRange(0, 60)
        self.room.setValue(15)
        room_row.addWidget(self.room)
        adjust_layout.addLayout(room_row)
        mic_row = QHBoxLayout()
        label = QLabel("Microphone")
        label.setMinimumWidth(115)
        mic_row.addWidget(label)
        self.microphone = QComboBox()
        self.microphone.setAccessibleName("Microphone")
        mic_row.addWidget(self.microphone, 1)
        self.refresh_button = QPushButton("Refresh")
        mic_row.addWidget(self.refresh_button)
        adjust_layout.addLayout(mic_row)
        self.reset_button = QPushButton("Reset sound tweaks")
        self.reset_button.setObjectName("link")
        adjust_layout.addWidget(self.reset_button, 0, Qt.AlignmentFlag.AlignRight)
        self.adjustments.hide()
        sound_layout.addWidget(self.adjustments)

        self.message = QLabel()
        self.message.setObjectName("message")
        self.message.setWordWrap(True)
        self.message.hide()
        layout.addWidget(self.message)
        layout.addStretch()
        footer_panel = QWidget()
        footer_panel.setObjectName("footer")
        footer_layout = QVBoxLayout(footer_panel)
        footer_layout.setContentsMargins(34, 0, 34, 18)
        footer_layout.setSpacing(9)
        outer.addWidget(footer_panel)
        footer = QHBoxLayout()
        self.status = QLabel("Ready to record")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.save_button = QToolButton()
        self.save_button.setText("Save…")
        self.save_button.setAccessibleName("Save music and original recording")
        self.save_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.save_button)
        self.save_both_action = menu.addAction("Both audio files…", self.save_both)
        self.save_instrument_action = menu.addAction("Instrument audio (WAV)…", self.save_audio)
        self.save_original_action = menu.addAction("Original recording (WAV)…", self.save_original)
        menu.addSeparator()
        self.save_notes_action = menu.addAction("Editable notes (MIDI)…", self.save_notes)
        self.save_button.setMenu(menu)
        footer.addWidget(self.save_button)
        footer_layout.addLayout(footer)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.hide()
        footer_layout.addWidget(self.progress)
        library_row = QHBoxLayout()
        local = QLabel("Save both keeps the original and instrument audio together in the folder you choose.")
        local.setObjectName("footnote")
        local.setWordWrap(True)
        library_row.addWidget(local, 1)
        self.library_button = QPushButton("Open Sounds folder")
        self.library_button.setObjectName("link")
        self.library_button.clicked.connect(self.open_sound_library)
        library_row.addWidget(self.library_button)
        footer_layout.addLayout(library_row)

    @staticmethod
    def _card():
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(22, 20, 22, 19)
        layout.setSpacing(12)
        return frame, layout

    @staticmethod
    def _heading(text):
        label = QLabel(text)
        label.setObjectName("heading")
        return label

    def _connect_controls(self):
        self.record_button.clicked.connect(self.toggle_recording)
        self.listen_button.clicked.connect(self.toggle_listen)
        self.original_button.clicked.connect(self.play_original)
        self.import_button.clicked.connect(self.open_recording)
        self.example_button.clicked.connect(lambda: self.load_recording(APP.parent / "examples" / "melody.wav"))
        self.adjust_button.toggled.connect(self.toggle_adjustments)
        self.refresh_button.clicked.connect(self._refresh_devices)
        self.reset_button.clicked.connect(self.reset_tweaks)
        self.instrument.currentIndexChanged.connect(self.sound_changed)
        self.speed.valueChanged.connect(self.sound_changed)
        self.octave.currentIndexChanged.connect(self.sound_changed)
        self.room.valueChanged.connect(self.sound_changed)
        self.volume.valueChanged.connect(self._volume_changed)
        self.microphone.currentIndexChanged.connect(self._save_preferences)
        self.input_mode.currentIndexChanged.connect(self.input_mode_changed)

    def settings(self):
        percussion = self.analysis_mode == "percussion"
        return {"patch": 0 if percussion else self.instrument.currentData(), "speed": self.speed.value() / 100,
                "octave": 0 if percussion else self.octave.currentData(), "room": self.room.value() / 100,
                "percussion": percussion, "drum_note": self.instrument.currentData() if percussion else self.drum_note}

    def _remember_sound(self):
        value = self.instrument.currentData()
        if value is not None:
            if self.analysis_mode == "percussion":
                self.drum_note = value
            else:
                self.melody_patch = value

    def _set_sound_list(self, mode, remember=True):
        if remember:
            self._remember_sound()
        self.analysis_mode = mode
        self.instrument.blockSignals(True)
        self.instrument.clear()
        for name, value in DRUMS if mode == "percussion" else INSTRUMENTS:
            self.instrument.addItem(name, value)
        index = self.instrument.findData(self.drum_note if mode == "percussion" else self.melody_patch)
        self.instrument.setCurrentIndex(max(0, index))
        self.instrument.blockSignals(False)
        self.octave.setEnabled(mode != "percussion")

    def input_mode_changed(self):
        self._save_preferences()
        if self.source and not self.job and not self.recorder.stream:
            self.load_recording(self.source)
        elif not self.source:
            self._set_sound_list("percussion" if self.input_mode.currentData() == "percussion" else "melody")
            self._update_buttons()

    def toggle_adjustments(self, shown):
        self.adjustments.setVisible(shown)
        self.adjust_button.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def _refresh_devices(self):
        old = self.microphone.currentText()
        self.microphone.blockSignals(True)
        self.microphone.clear()
        self.microphone.addItem("System default", None)
        try:
            devices, hosts = sd.query_devices(), sd.query_hostapis()
            preferred = [i for i,d in enumerate(devices) if d["max_input_channels"] > 0 and "WASAPI" in hosts[d["hostapi"]]["name"]]
            candidates = preferred or [i for i,d in enumerate(devices) if d["max_input_channels"] > 0]
            seen = set()
            for index in candidates:
                name = devices[index]["name"]
                if name not in seen:
                    self.microphone.addItem(name, index)
                    seen.add(name)
            chosen = self.microphone.findText(old or getattr(self, "preferred_microphone", "System default"))
            self.microphone.setCurrentIndex(max(0, chosen))
        except Exception:
            LOG.exception("Could not enumerate microphones")
        self.microphone.blockSignals(False)

    def toggle_recording(self):
        if self.recorder.stream:
            audio, rate = self.recorder.stop()
            self.waveform.live = False
            if len(audio) < rate * .25:
                self._error("Record a few notes or taps before you stop recording.")
                self._update_buttons()
                return
            name = datetime.now().strftime("Humming-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:5] + ".wav"
            path = ROOT / "Recordings" / name
            sf.write(str(path), audio, rate, subtype="PCM_16")
            self._analyze(path, audio, rate)
            return
        if self.job:
            return
        self.stop_playback()
        self.message.hide()
        try:
            self.recorder.start(self.microphone.currentData())
        except Exception:
            LOG.exception("Microphone could not start")
            self._error("I couldn't open the microphone. Under Adjust sound, choose your microphone and try again. If Windows asks, allow microphone access for this app.")
            return
        self.waveform.has_audio = False
        self.waveform.peaks[:] = 0
        self.waveform.live = True
        self.status.setText("Recording — hum or tap, then press Stop recording")
        self._update_buttons()

    def open_recording(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open a recording", str(ROOT / "Recordings"),
                    "Audio recordings (*.wav *.flac *.ogg *.mp3);;All files (*)")
        if path:
            self.load_recording(Path(path))

    def load_recording(self, path):
        if self.job or self.recorder.stream:
            return
        try:
            audio, rate = read_audio(path)
            self._analyze(Path(path), audio, rate)
        except Exception as error:
            LOG.exception("Could not open recording")
            self._error(str(error))

    def _analyze(self, path, audio, rate, autoplay=True):
        self.stop_playback()
        self.debounce.stop()
        self.source = Path(path)
        self.notes, self.rendered = [], None
        self.notes_file = ROOT / "Sessions" / ("Melody-" + uuid.uuid4().hex + ".json")
        self.waveform.set_audio(audio)
        self.duration_label.setText(f"{len(audio) / rate:.1f} seconds recorded")
        self.message.hide()
        write_json(STATE, {"source": str(self.source), "notes_file": str(self.notes_file)})
        self._start_job({"kind": "analyze", "input": str(path), "output": str(self.notes_file),
                         "mode": self.input_mode.currentData()},
                        {"kind": "analyze", "autoplay": autoplay}, "Finding your notes or taps…")

    def _start_job(self, specification, context, text):
        if self.job:
            return
        worker = AudioJob(specification, context, self)
        self.job = worker
        self.jobs.append(worker)
        worker.completed.connect(self._job_completed)
        worker.finished.connect(lambda: self.jobs.remove(worker) if worker in self.jobs else None)
        worker.finished.connect(worker.deleteLater)
        self.status.setText(text)
        self.progress.show()
        self._update_buttons()
        worker.start()

    def _job_completed(self, response):
        if self.closing:
            return
        context = response["context"]
        self.job = None
        self.progress.hide()
        if not response["ok"]:
            self._error(response.get("error", "I couldn't finish that melody. Please try again."))
            self._update_buttons()
            return
        if context["kind"] == "analyze":
            self.notes = [Note(**n) for n in response["result"]["notes"]]
            self._set_sound_list(response["result"].get("mode", "melody"))
            self._save_preferences()
            write_json(STATE, {"source": str(self.source), "notes_file": str(self.notes_file)})
            self._start_render(context.get("autoplay", True))
            return
        if context["revision"] != self.revision:
            self._start_render(True)
            return
        try:
            self.rendered, rate = sf.read(response["result"]["audio"], dtype="float32", always_2d=True)
            if rate != SAMPLE_RATE:
                raise ValueError("Unexpected playback sample rate")
        except Exception:
            LOG.exception("Could not read rendered instrument")
            self._error("I couldn't prepare playback. Try choosing the instrument again.")
            self._update_buttons()
            return
        unit = "drum hits" if self.analysis_mode == "percussion" else "notes"
        self.sound_hint.setText(f"{len(self.notes)} {unit} captured. Changing the sound plays it again.")
        self.status.setText("Ready to listen")
        self._update_buttons()
        if context.get("autoplay"):
            self.toggle_listen()

    def _start_render(self, autoplay=False):
        if not self.notes or not self.notes_file or self.recorder.stream or self.job:
            return
        self.debounce.stop()
        output = CACHE / ("sound-" + uuid.uuid4().hex + ".wav")
        self._start_job({"kind": "render", "notes": str(self.notes_file), "settings": self.settings(), "output": str(output)},
                        {"kind": "render", "revision": self.revision, "autoplay": autoplay},
                        f"Preparing {self.instrument.currentText().lower()}…")

    def sound_changed(self, *_):
        self.revision += 1
        self.rendered = None
        self.stop_playback()
        self.speed_label.setText(f"Speed {self.speed.value()}%")
        self.room_label.setText(f"Room {self.room.value()}%")
        self._save_preferences()
        self._update_buttons()
        if self.notes and not self.recorder.stream:
            self.debounce.start()

    def reset_tweaks(self):
        self.speed.setValue(100)
        self.octave.setCurrentIndex(2)
        self.room.setValue(15)

    def _volume_changed(self, *_):
        self._save_preferences()
        if self.playing and self.play_buffer is not None and self.play_source is not None:
            # sounddevice reads this same float buffer. Update future samples
            # without restarting the melody when the volume slider is moved.
            np.multiply(self.play_source, self.volume.value() / 100, out=self.play_buffer)

    def _play(self, audio, rate, kind):
        self.stop_playback()
        try:
            self.play_source = np.ascontiguousarray(audio, dtype=np.float32)
            self.play_buffer = np.ascontiguousarray(audio * (self.volume.value() / 100), dtype=np.float32)
            sd.play(self.play_buffer, rate, blocking=False)
            self.playing = kind
            self.status.setText("Playing your original recording" if kind == "original" else f"Playing {self.instrument.currentText().lower()}")
            self._update_buttons()
        except Exception:
            LOG.exception("Audio playback failed")
            self._error("I couldn't play sound. Check your headphones or speakers, then press Listen again.")

    def toggle_listen(self):
        if self.playing == "instrument":
            self.stop_playback()
        elif self.rendered is not None and not self.job:
            self._play(self.rendered, SAMPLE_RATE, "instrument")

    def play_original(self):
        if self.playing == "original":
            self.stop_playback()
        elif self.source and not self.recorder.stream:
            try:
                audio, rate = read_audio(self.source)
                self._play(audio, rate, "original")
            except Exception as error:
                self._error(str(error))

    def stop_playback(self):
        sd.stop()
        self.playing = None
        self.play_buffer = None
        self.play_source = None
        if self.notes and not self.job:
            self.status.setText("Ready to listen")
        self._update_buttons()

    def _tick(self):
        if self.recorder.stream:
            elapsed = time.monotonic() - self.recorder.started
            self.duration_label.setText(f"Recording   {int(elapsed) // 60:02}:{int(elapsed) % 60:02}")
            self.waveform.live_level(self.recorder.peak)
            if self.recorder.peak >= .98:
                self.status.setText("The sound is loud — move the microphone a little farther away")
            else:
                self.status.setText("Recording — hum or tap, then press Stop recording")
            if elapsed >= MAX_SECONDS:
                self.toggle_recording()
        if self.playing:
            try:
                active = sd.get_stream().active
            except Exception:
                active = False
            if not active:
                self.playing = None
                self.play_buffer = None
                self.play_source = None
                self.status.setText("Ready to listen")
                self._update_buttons()

    def _update_buttons(self):
        recording = bool(self.recorder.stream)
        busy = self.job is not None
        self.record_button.setEnabled(not busy)
        label = "Record taps" if self.input_mode.currentData() == "percussion" else "Record sound"
        self.record_button.setText("■   Stop recording" if recording else "●   Record again" if self.source else "●   " + label)
        self.record_button.setAccessibleName("Stop recording" if recording else label)
        self.input_mode.setEnabled(not busy and not recording)
        self.import_button.setEnabled(not busy and not recording)
        self.example_button.setEnabled(not busy and not recording)
        self.original_button.setEnabled(self.source is not None and not recording and not busy)
        self.original_button.setText("■   Stop original" if self.playing == "original" else "Hear original")
        self.listen_button.setEnabled(self.rendered is not None and not busy and not recording)
        self.listen_button.setText("■   Stop" if self.playing == "instrument" else "▶   Listen")
        can_save = self.source is not None and not busy and not recording
        has_sound = self.rendered is not None and can_save
        self.save_button.setEnabled(can_save)
        self.save_both_action.setEnabled(has_sound)
        self.save_instrument_action.setEnabled(has_sound)
        self.save_original_action.setEnabled(can_save)
        self.save_notes_action.setEnabled(bool(self.notes) and can_save)
        self.microphone.setEnabled(not recording)
        self.refresh_button.setEnabled(not recording)

    def _error(self, text):
        self.message.setText(text)
        self.message.show()
        self.status.setText("Ready to try again")

    def open_sound_library(self):
        try:
            SOUND_LIBRARY.root.mkdir(parents=True, exist_ok=True)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(SOUND_LIBRARY.root))):
                raise OSError("File Explorer could not open the Sounds folder")
        except OSError:
            LOG.exception("Could not open the Sounds library")
            self._error("I couldn't open your Sounds folder. Your recordings are still safe in Hum to Tune.")

    def _mirror_saved(self, files):
        try:
            for path, kind in files:
                copied = SOUND_LIBRARY.copy(path, kind)
                LOG.info("Copied to Sounds: %s", copied)
            return True
        except (OSError, ValueError):
            LOG.exception("Could not copy saved audio or notes to Sounds")
            self._error("Your files were saved, but I couldn't copy them to your Sounds folder. Check that the folder is available.")
            return False

    def _default_save_path(self, kind, name):
        folder = SOUND_LIBRARY.root / SOUND_LIBRARY.FOLDERS[kind]
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            LOG.exception("The default Sounds save folder is unavailable")
            folder = ROOT / "Exports"
        return str(folder / name)

    def save_audio(self):
        if self.rendered is None:
            return
        name = ("My rhythm" if self.analysis_mode == "percussion" else "My melody") + " - " + self.instrument.currentText() + ".wav"
        path, _ = QFileDialog.getSaveFileName(self, "Save your music", self._default_save_path("instrument", name), "WAV audio (*.wav)")
        if path:
            try:
                destination = Path(path)
                if destination.suffix.lower() != ".wav":
                    destination = destination.with_suffix(".wav")
                if self.source and destination.resolve() == self.source.resolve():
                    self._error("Choose a different filename to keep your original humming safe.")
                    return
                sf.write(str(destination), self.rendered * (self.volume.value() / 100), SAMPLE_RATE, subtype="PCM_16")
                if self._mirror_saved([(destination, "instrument")]):
                    self.status.setText(f"Saved {destination.name} · copied to Sounds")
            except Exception:
                LOG.exception("Audio save failed")
                self._error("I couldn't save there. Choose another folder or filename.")

    def _copy_original(self, destination):
        if self.source.resolve() == destination.resolve():
            return
        if self.source.suffix.lower() == ".wav":
            # Keep recorded WAVs byte-for-byte intact, including every channel.
            shutil.copyfile(self.source, destination)
        else:
            audio, rate = sf.read(str(self.source), dtype="float32", always_2d=True)
            sf.write(str(destination), audio, rate, subtype="FLOAT", format="WAV")

    def save_original(self):
        if not self.source:
            return
        name = "My rhythm - original recording.wav" if self.analysis_mode == "percussion" else "My melody - original hum.wav"
        path, _ = QFileDialog.getSaveFileName(self, "Save your original recording",
                    self._default_save_path("original", name), "WAV audio (*.wav)")
        if path:
            try:
                destination = Path(path)
                if destination.suffix.lower() != ".wav":
                    destination = destination.with_suffix(".wav")
                self._copy_original(destination)
                if self._mirror_saved([(destination, "original")]):
                    self.status.setText(f"Saved {destination.name} · copied to Sounds")
            except Exception:
                LOG.exception("Original recording save failed")
                self._error("I couldn't save the humming there. Choose another folder or filename.")

    def save_both(self):
        if not self.source or self.rendered is None:
            return
        default_name = "My rhythm" if self.analysis_mode == "percussion" else "My melody"
        path, _ = QFileDialog.getSaveFileName(self, "Choose a name and folder for both recordings",
                    self._default_save_path("instrument", default_name), "Recording name (*)")
        if not path:
            return
        base = Path(path)
        name = base.stem if base.suffix.lower() == ".wav" else base.name
        original_kind = "original recording" if self.analysis_mode == "percussion" else "original hum"
        original_folder = base.parent
        original = original_folder / f"{name} - {original_kind}.wav"
        instrument = base.parent / f"{name} - {self.instrument.currentText()}.wav"
        if self.source.resolve() in (instrument.resolve(), original.resolve()):
            self._error("Choose a different filename to keep your original humming safe.")
            return
        existing = [p.name for p in (original, instrument) if p.exists()]
        if existing:
            answer = QMessageBox.question(self, "Replace saved recordings?",
                "These files already exist:\n\n" + "\n".join(existing) + "\n\nReplace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            # Finish both files before putting either into the chosen folder.
            original_folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".hum-to-music-", dir=base.parent) as temporary:
                staged_original = Path(temporary) / "original.wav"
                staged_instrument = Path(temporary) / "instrument.wav"
                self._copy_original(staged_original)
                sf.write(str(staged_instrument), self.rendered * (self.volume.value() / 100),
                         SAMPLE_RATE, subtype="PCM_16")
                staged_original.replace(original)
                staged_instrument.replace(instrument)
            self.status.setText("Saved original and instrument together")
        except Exception:
            LOG.exception("Saving both recordings failed")
            self._error("I couldn't save both recordings there. Choose another folder or filename.")

    def save_notes(self):
        if not self.notes:
            return
        name = "My rhythm.mid" if self.analysis_mode == "percussion" else "My melody.mid"
        path, _ = QFileDialog.getSaveFileName(self, "Save editable notes", self._default_save_path("instrument", name), "MIDI notes (*.mid)")
        if path:
            try:
                destination = Path(path)
                if destination.suffix.lower() != ".mid":
                    destination = destination.with_suffix(".mid")
                sound = self.settings()
                export_midi(destination, self.notes, sound["patch"], sound["speed"], sound["octave"],
                            percussion=sound["percussion"], drum_note=sound["drum_note"])
                self.status.setText(f"Saved {destination.name}")
            except Exception:
                LOG.exception("MIDI save failed")
                self._error("I couldn't save there. Choose another folder or filename.")

    def _save_preferences(self, *_):
        try:
            self._remember_sound()
            write_json(PREFERENCES, {**self.settings(), "volume": self.volume.value(), "microphone": self.microphone.currentText(),
                                    "input_mode": self.input_mode.currentData(), "melody_patch": self.melody_patch,
                                    "octave": self.octave.currentData()})
        except OSError:
            LOG.exception("Could not save preferences")

    def _load_preferences(self):
        try:
            settings = json.loads(PREFERENCES.read_text(encoding="utf-8"))
            self.melody_patch = int(settings.get("melody_patch", settings.get("patch", 0)))
            self.drum_note = int(settings.get("drum_note", 38))
            self.input_mode.setCurrentIndex(max(0, self.input_mode.findData(settings.get("input_mode", "auto"))))
            self._set_sound_list("percussion" if self.input_mode.currentData() == "percussion" else "melody", remember=False)
            self.speed.setValue(round(float(settings.get("speed", 1)) * 100))
            index = self.octave.findData(int(settings.get("octave", 0)))
            self.octave.setCurrentIndex(index if index >= 0 else 2)
            self.room.setValue(round(float(settings.get("room", .15)) * 100))
            self.volume.setValue(int(settings.get("volume", 75)))
            self.preferred_microphone = settings.get("microphone", "System default")
        except (OSError, ValueError, TypeError):
            pass
        self.speed_label.setText(f"Speed {self.speed.value()}%")
        self.room_label.setText(f"Room {self.room.value()}%")

    def _restore_session(self):
        try:
            saved = json.loads(STATE.read_text(encoding="utf-8"))
            notes_file = Path(saved["notes_file"])
            source = Path(saved["source"])
            audio, rate = read_audio(source)
            if not notes_file.exists():
                self._analyze(source, audio, rate, autoplay=False)
                return
            stored = json.loads(notes_file.read_text(encoding="utf-8"))
            self.source, self.notes_file = source, notes_file
            self.notes = [Note(**n) for n in stored["notes"]]
            self._set_sound_list(stored.get("mode", "melody"))
            self.waveform.set_audio(audio)
            self.duration_label.setText(f"{len(audio) / rate:.1f} seconds recorded")
            self._start_render(False)
        except Exception:
            LOG.info("Previous recording could not be restored; opening a fresh workspace")

    def closeEvent(self, event):
        self.closing = True
        self.timer.stop()
        self.debounce.stop()
        self.stop_playback()
        if self.recorder.stream:
            audio, rate = self.recorder.stop()
            if len(audio):
                filename = ROOT / "Recordings" / ("Humming-recovered-" + uuid.uuid4().hex[:10] + ".wav")
                sf.write(str(filename), audio, rate, subtype="PCM_16")
        self._save_preferences()
        for job in list(self.jobs):
            job.cancel()
            job.wait(3000)
        event.accept()


STYLE = """
QMainWindow, QWidget#shell, QWidget#footer, QScrollArea, QScrollArea > QWidget > QWidget { background: #edf2f0; }
QWidget { color: #243c3a; font-family: 'Segoe UI'; font-size: 14px; }
QLabel#title { font-size: 29px; font-weight: 700; letter-spacing: -0.5px; }
QLabel#subtitle { font-size: 15px; color: #667b77; margin-bottom: 3px; }
QFrame#card { background: #ffffff; border: 1px solid #dce6e2; border-radius: 14px; }
QLabel#heading { font-size: 17px; font-weight: 650; }
QLabel#muted { color: #73847f; font-size: 13px; }
QLabel#footnote { color: #80918b; font-size: 12px; }
QLabel#status { color: #26776f; font-size: 13px; }
QLabel#message { background: #fff0e5; color: #8b4d20; border-radius: 8px; padding: 12px; }
QPushButton, QToolButton { background: #f3f7f5; border: 1px solid #dbe4df; border-radius: 7px; padding: 8px 13px; }
QPushButton:hover, QToolButton:hover { background: #e6eeea; }
QPushButton:disabled, QToolButton:disabled { color: #a7b2ad; background: #f1f4f2; border-color: #e9eeeb; }
QPushButton#record { color: #ffffff; background: #bd5148; border: none; font-size: 16px; font-weight: 650; }
QPushButton#record:hover { background: #a84038; }
QPushButton#record:disabled { background: #d8ada7; }
QPushButton#listen { color: white; background: #247b70; border: none; font-weight: 650; font-size: 16px; }
QPushButton#listen:hover { background: #1c655d; }
QPushButton#listen:disabled { color: #a8bdb7; background: #e7efeb; }
QPushButton#link { background: transparent; color: #438477; border: none; padding: 5px; font-size: 12px; }
QPushButton#link:hover { background: #eef5f1; }
QComboBox { background: #f9fbfa; border: 1px solid #cfddd5; border-radius: 7px; padding: 8px 13px; font-size: 15px; }
QComboBox::drop-down { border: none; width: 27px; }
QComboBox QAbstractItemView { background: #ffffff; selection-background-color: #dceee5; selection-color: #243c3a; padding: 4px; }
QSlider::groove:horizontal { height: 5px; background: #dce7e1; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #5b9b89; border-radius: 2px; }
QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #2e7e6d; border-radius: 7px; }
QProgressBar { background: #e1ece6; border: none; border-radius: 2px; }
QProgressBar::chunk { background: #5eaa92; }
QMenu { background: white; border: 1px solid #d3dfd7; padding: 5px; }
QMenu::item { padding: 8px 16px; }
QMenu::item:selected { background: #e2efe8; }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", type=Path)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Hum to Tune")
    app.setOrganizationName("HumToMusic")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    server = QLocalServer(app)
    server_name = "hum-to-melody-" + hashlib.sha256(str(DATA).casefold().encode()).hexdigest()[:16]
    if not server.listen(server_name):
        client = QLocalSocket(app)
        client.connectToServer(server_name)
        if client.waitForConnected(400):
            client.write(b"activate")
            client.waitForBytesWritten(200)
            return 0
        QLocalServer.removeServer(server_name)
        if not server.listen(server_name):
            LOG.error("Could not acquire the local app instance")
            return 1
    window = HumToMusic(restore=not (args.fresh or args.open))

    def activate_existing():
        connection = server.nextPendingConnection()
        if connection:
            connection.disconnectFromServer()
            connection.deleteLater()
        window.showNormal()
        window.raise_()
        window.activateWindow()

    server.newConnection.connect(activate_existing)
    window.show()
    if args.open:
        QTimer.singleShot(200, lambda: window.load_recording(args.open))
    LOG.info("Application opened")
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        LOG.exception("Application failed")
        raise
