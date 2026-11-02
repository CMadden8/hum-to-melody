"""Integration checks against the real local analysis and synthesis engines."""
import json
from pathlib import Path
import time

import mido
import numpy as np
import pytest
import soundfile as sf

from engine import (DRUMS, InstrumentPlayer, MelodyError, Note, ROOT, adjusted_notes,
                    analyze_recording, detect_taps, export_midi, read_audio, transcribe)


@pytest.fixture(scope="module")
def detected():
    audio, rate = read_audio(ROOT / "examples" / "melody.wav")
    return transcribe(audio, rate)


def test_native_detector_recovers_pitch_rests_and_repeated_notes(detected):
    expected = json.loads((ROOT / "examples" / "expected-notes.json").read_text())["notes"]
    assert [n.pitch for n in detected] == [n["midi"] for n in expected]
    assert max(abs(a.start - b["start_seconds"]) for a,b in zip(detected, expected)) < .05
    assert max(abs(a.duration - b["duration_seconds"]) for a,b in zip(detected, expected)) < .05


def test_silence_returns_an_actionable_error():
    with pytest.raises(MelodyError, match="couldn't hear a melody"):
        transcribe(np.zeros(44100, dtype=np.float32), 44100)


def test_stereo_import_does_not_cancel_out_opposite_channels(tmp_path):
    audio, rate = read_audio(ROOT / "examples" / "melody.wav")
    path = tmp_path / "opposite.wav"
    sf.write(path, np.column_stack([audio, -audio]), rate, subtype="FLOAT")
    imported, actual_rate = read_audio(path)
    assert actual_rate == rate
    assert np.max(np.abs(imported)) > .1


def test_instrument_changes_and_speed_render_real_audio(detected):
    with InstrumentPlayer() as player:
        piano = player.render(detected, patch=0, room=0)
        strings = player.render(detected, patch=48, room=0)
        brass = player.render(detected, patch=61, room=0)
        faster = player.render(detected, patch=0, speed=1.5, octave=1, room=0)
    for audio in (piano, strings, brass, faster):
        assert audio.shape[1] == 2
        assert np.isfinite(audio).all()
        assert .7 < np.max(np.abs(audio)) < .9
        assert np.sqrt(np.mean(audio ** 2)) > .005
    assert not np.allclose(piano[:100000], strings[:100000], atol=.01)
    assert not np.allclose(brass[:100000], strings[:100000], atol=.01)
    assert len(faster) < len(piano) * .8


def test_midi_export_preserves_adjusted_notes_and_timing(tmp_path, detected):
    path = tmp_path / "melody.mid"
    export_midi(path, detected, patch=48, speed=1.25, octave=-1)
    messages = list(mido.MidiFile(path))
    actual = []
    elapsed = 0
    for message in messages:
        elapsed += message.time
        if message.type == "note_on" and message.velocity:
            actual.append((message.note, elapsed))
    expected = adjusted_notes(detected, 1.25, -1)
    assert [pitch for pitch,_ in actual] == [n.pitch for n in expected]
    assert max(abs(t - n.start) for (_,t),n in zip(actual, expected)) < .001
    assert any(m.type == "program_change" and m.program == 48 for m in messages)


def wait_until(app, condition, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(.015)
    raise AssertionError("The app did not finish within the expected time")


def test_record_stop_instrument_change_and_session_restore(monkeypatch, tmp_path):
    # Real analysis/rendering processes, with deterministic audio at the mic boundary.
    import app as ui
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    qt = QApplication.instance() or QApplication([])
    qt.setStyleSheet(ui.STYLE)
    monkeypatch.setattr(ui, "STATE", tmp_path / "last-session.json")
    monkeypatch.setattr(ui, "PREFERENCES", tmp_path / "preferences.json")
    # Tests must never create example takes in the user's real recordings/library.
    test_root = tmp_path / "app-data"
    for folder in ("Recordings", "Sessions", "Exports", "cache"):
        (test_root / folder).mkdir(parents=True)
    monkeypatch.setattr(ui, "ROOT", test_root)
    monkeypatch.setattr(ui, "CACHE", test_root / "cache")
    library = ui.SoundLibrary(tmp_path / "game-sounds")
    monkeypatch.setattr(ui, "SOUND_LIBRARY", library)
    window = ui.HumToMusic(restore=False)
    window.show()
    qt.processEvents()
    footer_y = window.save_button.mapTo(window, window.save_button.rect().center()).y()
    assert 0 < footer_y < window.height()
    window.adjust_button.setChecked(True)
    qt.processEvents()
    assert window.save_button.mapTo(window, window.save_button.rect().center()).y() == footer_y
    window.adjust_button.setChecked(False)
    audio, rate = read_audio(ROOT / "examples" / "melody.wav")
    played = []
    monkeypatch.setattr(window, "_play", lambda data,sr,kind: played.append((data.shape,sr,kind)))

    def start(_device):
        window.recorder.stream = object()
        window.recorder.started = time.monotonic()

    def stop():
        window.recorder.stream = None
        return audio, rate

    monkeypatch.setattr(window.recorder, "start", start)
    monkeypatch.setattr(window.recorder, "stop", stop)
    try:
        QTest.mouseClick(window.record_button, Qt.MouseButton.LeftButton)
        assert window.recorder.stream is not None
        assert "Stop" in window.record_button.text()
        QTest.mouseClick(window.record_button, Qt.MouseButton.LeftButton)
        wait_until(qt, lambda: window.rendered is not None and window.job is None)
        assert len(window.notes) == 12
        assert window.listen_button.isEnabled() and window.save_button.isEnabled()
        assert window.source.exists() and ui.STATE.exists()
        assert not library.root.exists(), "Unsaved takes must not fill the game sound folders"
        piano = window.rendered.copy()
        assert played[-1][2] == "instrument"
        window.instrument.setCurrentIndex(window.instrument.findData(48))
        wait_until(qt, lambda: window.rendered is not None and window.job is None)
        assert window.settings()["patch"] == 48
        assert not np.allclose(piano[:100000], window.rendered[:100000], atol=.01)
        assert len(played) == 2
        audio_export = tmp_path / "My melody.wav"
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a,**_k: (str(audio_export), "WAV audio (*.wav)"))
        window.save_audio()
        exported, export_rate = sf.read(audio_export, always_2d=True)
        assert export_rate == 48000 and exported.shape == window.rendered.shape
        assert np.max(np.abs(exported)) < 1
        assert np.sqrt(np.mean(exported**2)) > .005
        assert (library.root / "Instruments" / audio_export.name).read_bytes() == audio_export.read_bytes()
        source_bytes = window.source.read_bytes()
        original_export = tmp_path / "Original humming.wav"
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a,**_k: (str(original_export), "WAV audio (*.wav)"))
        window.save_original()
        assert original_export.read_bytes() == source_bytes
        assert (library.root / "Human voice" / original_export.name).read_bytes() == source_bytes
        pair_base = tmp_path / "My tune"
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a,**_k: (str(pair_base), "Recording name (*)"))
        window.save_both()
        assert (tmp_path / "My tune - original hum.wav").read_bytes() == source_bytes
        paired, paired_rate = sf.read(tmp_path / "My tune - Strings.wav", always_2d=True)
        assert paired_rate == export_rate
        assert np.array_equal(paired, exported)
        assert not (library.root / "Human voice" / "My tune - original hum.wav").exists()
        assert not (library.root / "Instruments" / "My tune - Strings.wav").exists()
        default_pair = library.root / "Instruments" / "Together in default folder"
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a,**_k: (str(default_pair), "Recording name (*)"))
        window.save_both()
        assert default_pair.with_name(default_pair.name + " - original hum.wav").read_bytes() == source_bytes
        assert default_pair.with_name(default_pair.name + " - Strings.wav").exists()
        assert not (library.root / "Human voice" / "Together in default folder - original hum.wav").exists()
        assert window.source.read_bytes() == source_bytes
        real_source = window.source
        source_at_risk = tmp_path / "Keep my original - Strings.wav"
        source_at_risk.write_bytes(source_bytes)
        window.source = source_at_risk
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a,**_k: (str(tmp_path / "Keep my original"), "Recording name (*)"))
        window.save_both()
        assert source_at_risk.read_bytes() == source_bytes
        assert not (tmp_path / "Keep my original - original hum.wav").exists()
        window.source = real_source
        notes_export = tmp_path / "My melody.mid"
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a,**_k: (str(notes_export), "MIDI notes (*.mid)"))
        window.save_notes()
        exported_notes = [m.note for m in mido.MidiFile(notes_export) if m.type == "note_on" and m.velocity]
        assert exported_notes == [n.pitch for n in window.notes]
        assert not (library.root / "MIDI").exists()
        assert "copied to Sounds" not in window.status.text()
        assert {p.name for p in library.root.iterdir()} == {"Instruments", "Human voice"}
        # A missing/unwritable secondary destination must preserve the main export
        # and explicitly report the failed additional copy.
        blocked = tmp_path / "blocked-library"
        blocked.write_text("a file is in the way")
        monkeypatch.setattr(ui, "SOUND_LIBRARY", ui.SoundLibrary(blocked))
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a,**_k: (str(audio_export), "WAV audio (*.wav)"))
        window.save_audio()
        assert audio_export.exists() and original_export.read_bytes() == source_bytes
        assert "couldn't copy" in window.message.text()
        assert "copied to Sounds" not in window.status.text()
        window._restore_session()
        wait_until(qt, lambda: window.job is None)
        assert len(window.notes) == 12 and window.rendered is not None
        assert len(played) == 2, "Restoring a session should not play unexpectedly"
        window.play_source = window.rendered
        window.play_buffer = window.rendered.copy()
        window.playing = "instrument"
        window.volume.setValue(35)
        assert len(played) == 2, "Changing volume should not restart the tune"
        assert np.allclose(window.play_buffer, window.play_source * .35)
    finally:
        window.close()
        qt.processEvents()


def test_game_library_preserves_collisions_and_skips_identical_copies(tmp_path):
    from game_library import SoundLibrary
    source = tmp_path / "My melody - Flute.wav"
    source.write_bytes(b"first take")
    library = SoundLibrary(tmp_path / "Sounds")
    first = library.copy(source, "instrument")
    assert first.read_bytes() == b"first take"
    assert library.copy(source, "instrument") == first
    source.write_bytes(b"second take")
    second = library.copy(source, "instrument")
    assert second.name == "My melody - Flute (2).wav"
    assert first.read_bytes() == b"first take"
    assert second.read_bytes() == source.read_bytes() == b"second take"
    assert library.copy(source, "instrument") == second
    assert len(list((library.root / "Instruments").iterdir())) == 2


def test_interrupted_game_library_copy_leaves_source_and_existing_takes_safe(monkeypatch, tmp_path):
    import game_library
    source = tmp_path / "Humming.wav"
    source.write_bytes(b"keep this original")
    library = game_library.SoundLibrary(tmp_path / "Sounds")

    def fail_copy(_source, target):
        target.write(b"incomplete")
        raise OSError("simulated disk error")

    monkeypatch.setattr(game_library.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="simulated disk error"):
        library.copy(source, "original")
    assert source.read_bytes() == b"keep this original"
    assert not list((library.root / "Human voice").iterdir())


@pytest.fixture
def tapped_rhythm():
    rate = 48000
    times = [0., .35, .78, .90, 1.55, 2.20]
    strengths = [1., .5, .9, .3, .7, .8]
    rng = np.random.default_rng(7)
    audio = rng.normal(0, .00007, round(2.6 * rate)).astype(np.float32)
    for start, strength in zip(times, strengths):
        length = round(.16 * rate)
        t = np.arange(length) / rate
        attack = rng.normal(0, .25, length) * np.exp(-t / .007)
        ring = .25 * np.sin(2 * np.pi * 210 * t) * np.exp(-t / .012)
        frame = round(start * rate)
        audio[frame:frame + length] += (attack + ring) * strength
    return audio, rate, times


def test_tap_detector_retains_irregular_rhythm_and_soft_quick_hits(tapped_rhythm):
    audio, rate, expected = tapped_rhythm
    hits = detect_taps(audio, rate)
    assert len(hits) == len(expected)
    assert max(abs(hit.start - start) for hit, start in zip(hits, expected)) < .015
    assert hits[0].velocity > hits[1].velocity > hits[3].velocity
    with pytest.raises(MelodyError, match="distinct taps"):
        detect_taps(np.zeros(rate), rate)
    with pytest.raises(MelodyError, match="distinct taps"):
        detect_taps(np.random.default_rng(4).normal(0, .005, rate * 3), rate)


def test_automatic_tap_fallback_and_real_drum_samples_and_midi(tapped_rhythm, tmp_path):
    audio, rate, times = tapped_rhythm
    hits, mode = analyze_recording(audio, rate, "auto")
    assert mode == "percussion" and len(hits) == len(times)
    with InstrumentPlayer() as player:
        sounds = [player.render(hits, percussion=True, drum_note=note, room=0) for _, note in DRUMS]
    for sound in sounds:
        assert sound.shape[1] == 2 and np.isfinite(sound).all()
        assert np.max(np.abs(sound)) > .7
    assert not np.allclose(sounds[0][:100000], sounds[1][:100000], atol=.01)
    path = tmp_path / "Rhythm.mid"
    export_midi(path, hits, speed=1.25, octave=2, percussion=True, drum_note=36)
    elapsed, onsets = 0., []
    for message in mido.MidiFile(path):
        elapsed += message.time
        if message.type == "note_on" and message.velocity:
            assert message.channel == 9 and message.note == 36
            onsets.append(elapsed)
    assert len(onsets) == len(times)
    assert max(abs((actual - onsets[0]) - (start - times[0]) / 1.25)
               for actual, start in zip(onsets, times)) < .015


def test_tapping_ui_reprocesses_existing_take_and_restores_percussion(monkeypatch, tmp_path, tapped_rhythm):
    import app as ui
    from PySide6.QtWidgets import QApplication
    qt = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, "STATE", tmp_path / "last-session.json")
    monkeypatch.setattr(ui, "PREFERENCES", tmp_path / "preferences.json")
    test_root = tmp_path / "app-data"
    for folder in ("Recordings", "Sessions", "Exports", "cache"):
        (test_root / folder).mkdir(parents=True)
    monkeypatch.setattr(ui, "ROOT", test_root)
    monkeypatch.setattr(ui, "CACHE", test_root / "cache")
    library = ui.SoundLibrary(tmp_path / "game-sounds")
    monkeypatch.setattr(ui, "SOUND_LIBRARY", library)
    audio, rate, times = tapped_rhythm
    source = tmp_path / "Desk taps.wav"
    sf.write(source, audio, rate, subtype="PCM_16")
    original_bytes = source.read_bytes()
    window = ui.HumToMusic(restore=False)
    played = []
    monkeypatch.setattr(window, "_play", lambda *_: played.append(True))
    try:
        window.load_recording(source)
        wait_until(qt, lambda: window.rendered is not None and window.job is None)
        assert window.analysis_mode == "percussion" and len(window.notes) == len(times)
        assert window.instrument.currentText() == "Snare drum"
        assert "drum hits" in window.sound_hint.text()
        assert not window.octave.isEnabled()
        assert played and not library.root.exists()
        window.instrument.setCurrentIndex(window.instrument.findData(36))
        wait_until(qt, lambda: window.rendered is not None and window.job is None)
        window.input_mode.setCurrentIndex(window.input_mode.findData("percussion"))
        wait_until(qt, lambda: window.rendered is not None and window.job is None)
        assert window.source == source and window.settings()["drum_note"] == 36
        assert len(window.notes) == len(times)
        base = tmp_path / "Saved rhythm"
        monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(base), ""))
        window.save_both()
        assert (tmp_path / "Saved rhythm - Kick drum.wav").exists()
        assert (tmp_path / "Saved rhythm - original recording.wav").read_bytes() == original_bytes
        assert not (library.root / "Human voice").exists()
        assert source.read_bytes() == original_bytes
        window._restore_session()
        wait_until(qt, lambda: window.job is None)
        assert window.analysis_mode == "percussion" and window.settings()["drum_note"] == 36
        assert window.rendered is not None and not window.octave.isEnabled()
        # A failed analysis also retains the source across restarts for retry.
        saved = json.loads(ui.STATE.read_text())
        Path(saved["notes_file"]).unlink()
        window._restore_session()
        wait_until(qt, lambda: window.job is None)
        assert window.analysis_mode == "percussion" and len(window.notes) == len(times)
        assert {p.name for p in library.root.iterdir()} == {"Instruments"}
    finally:
        window.close()
        qt.processEvents()
