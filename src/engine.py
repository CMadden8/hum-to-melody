"""Local signal processing and sample playback for Hum to Tune.

The unmodified pYIN Vamp plugin and FluidSynth do the musical processing.
Neither the Sonic Visualiser nor LMMS application is launched by this module.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import ctypes as C
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import threading

import mido
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 48000
MAX_SECONDS = 180
INSTRUMENTS = [
    ("Grand piano", 0), ("Soft electric piano", 4), ("Strings", 48),
    ("Violin", 40), ("Cello", 42), ("Brass section", 61), ("Trumpet", 56),
    ("French horn", 60), ("Flute", 73), ("Clarinet", 71),
    ("Acoustic guitar", 24), ("Harp", 46), ("Marimba", 12),
    ("Celesta", 8), ("Organ", 19), ("Choir", 52),
    ("Synth lead", 80), ("Warm synth pad", 89),
]
DRUMS = [("Snare drum", 38), ("Kick drum", 36), ("Closed hi-hat", 42),
         ("Hand clap", 39), ("Low tom", 45), ("High tom", 50),
         ("Tambourine", 54), ("Cowbell", 56), ("Wood block", 76), ("Rim shot", 37)]


class MelodyError(Exception):
    """A recoverable error that can be shown directly to the musician."""


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    duration: float
    velocity: int = 90


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_audio(path: str | Path):
    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, sf.LibsndfileError) as error:
        raise MelodyError("I couldn't open that recording. Try a WAV, FLAC, OGG or MP3 file.") from error
    if len(audio) < rate * .25:
        raise MelodyError("That recording is very short. Record a few notes or taps and try again.")
    if len(audio) > rate * MAX_SECONDS:
        raise MelodyError("Use a recording shorter than three minutes for this version.")
    if not np.isfinite(audio).all():
        raise MelodyError("That audio file contains invalid samples. Try another recording.")
    if not 8000 <= rate <= 192000:
        raise MelodyError("That recording uses an unusual audio format. Try a WAV recording.")
    # Use the strongest channel rather than cancelling opposite-polarity stereo.
    strongest = int(np.argmax(np.mean(audio.astype(np.float64) ** 2, axis=0)))
    return np.ascontiguousarray(audio[:, strongest]), rate


P, U, I, F, S = C.c_void_p, C.c_uint, C.c_int, C.c_float, C.c_char_p


class _Output(C.Structure):
    _fields_ = [("identifier", S), ("name", S), ("description", S), ("unit", S),
                ("fixed_bins", I), ("bins", U), ("bin_names", P), ("extents", I),
                ("minimum", F), ("maximum", F), ("quantized", I), ("step", F),
                ("sample_type", I), ("sample_rate", F), ("has_duration", I)]


class _Feature(C.Structure):
    _fields_ = [("timestamp", I), ("sec", I), ("nsec", I), ("count", U),
                ("values", C.POINTER(F)), ("label", S)]


class _Duration(C.Structure):
    _fields_ = [("has_duration", I), ("sec", I), ("nsec", I)]


class _FeatureUnion(C.Union):
    _fields_ = [("v1", _Feature), ("v2", _Duration)]


class _FeatureList(C.Structure):
    _fields_ = [("count", U), ("features", C.POINTER(_FeatureUnion))]


_CALLBACKS = ["instantiate", "cleanup", "initialise", "reset", "get_parameter",
              "set_parameter", "get_program", "select_program", "preferred_step",
              "preferred_block", "minimum_channels", "maximum_channels", "output_count",
              "output_descriptor", "release_output", "process", "remaining", "release_features"]


class _Descriptor(C.Structure):
    _fields_ = [("api", U), ("identifier", S), ("name", S), ("description", S),
                ("maker", S), ("version", I), ("copyright", S), ("parameter_count", U),
                ("parameters", P), ("program_count", U), ("programs", P), ("domain", I)] + [(n, P) for n in _CALLBACKS]


def transcribe(audio: np.ndarray, rate: int, progress=None) -> list[Note]:
    signal = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    if len(signal) < rate * .25:
        raise MelodyError("Hum a few notes, then press Stop recording.")
    signal = signal - float(np.mean(signal))
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if peak < .003 or rms < .0004:
        raise MelodyError("I couldn't hear a melody. Move closer to the microphone and try again.")
    signal *= .8 / max(peak, .01)
    search = os.add_dll_directory(str(ROOT / "Runtime" / "pyin"))
    try:
        library = C.CDLL(str(ROOT / "Runtime" / "pyin" / "pyin.dll"))
        library.vampGetPluginDescriptor.argtypes = [U, U]
        library.vampGetPluginDescriptor.restype = C.POINTER(_Descriptor)
        pointer = library.vampGetPluginDescriptor(2, 0)
        if not pointer or pointer.contents.identifier != b"pyin":
            raise RuntimeError("The installed pYIN library could not be identified")
        descriptor = pointer.contents
        call = lambda name, result, *args: C.CFUNCTYPE(result, *args)(getattr(descriptor, name))
        handle = call("instantiate", P, C.POINTER(_Descriptor), F)(pointer, rate)
        if not handle:
            raise RuntimeError("pYIN could not create a processor")
        try:
            block = call("preferred_block", U, P)(handle) or 2048
            step = call("preferred_step", U, P)(handle) or block // 4
            if not call("initialise", I, P, U, U, U)(handle, 1, step, block):
                raise RuntimeError("pYIN could not initialise")
            note_index = None
            for index in range(call("output_count", U, P)(handle)):
                out_pointer = call("output_descriptor", C.POINTER(_Output), P, U)(handle, index)
                try:
                    if out_pointer.contents.identifier == b"notes":
                        note_index = index
                finally:
                    call("release_output", None, C.POINTER(_Output))(out_pointer)
            if note_index is None:
                raise RuntimeError("pYIN did not expose a note output")
            notes = []

            def consume(features):
                if not features:
                    return
                try:
                    result = features[note_index]
                    for index in range(result.count):
                        item = result.features[index].v1
                        duration = result.features[result.count + index].v2
                        if not item.count or not duration.has_duration:
                            continue
                        hz = float(item.values[0])
                        length = duration.sec + duration.nsec / 1e9
                        if not math.isfinite(hz) or not 45 <= hz <= 2100 or length < .075:
                            continue
                        pitch = round(69 + 12 * math.log2(hz / 440))
                        start = max(0., item.sec + item.nsec / 1e9)
                        notes.append(Note(pitch, start, length))
                finally:
                    call("release_features", None, C.POINTER(_FeatureList))(features)

            process = call("process", C.POINTER(_FeatureList), P, C.POINTER(C.POINTER(F)), I, I)
            padded = np.pad(signal, (0, block))
            for start in range(0, len(signal), step):
                values = padded[start:start + block]
                channels = (C.POINTER(F) * 1)(values.ctypes.data_as(C.POINTER(F)))
                timestamp = round(start / rate * 1e9)
                consume(process(handle, channels, timestamp // 10**9, timestamp % 10**9))
                if progress and start % (step * 64) == 0:
                    progress(int(start / len(signal) * 90))
            consume(call("remaining", C.POINTER(_FeatureList), P)(handle))
        finally:
            call("cleanup", None, P)(handle)
    finally:
        search.close()
    notes.sort(key=lambda n: n.start)
    if not notes:
        raise MelodyError("I couldn't find clear notes. Try humming a little more steadily, without background music.")
    return notes


def detect_taps(audio: np.ndarray, rate: int) -> list[Note]:
    """Detect percussive attacks from short-time energy, without pitch inference."""
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(signal) < rate * .25:
        raise MelodyError("Tap a short rhythm, then press Stop recording.")
    signal = signal - float(np.mean(signal))
    # Five-millisecond energy frames retain a tap's timing without needing
    # a pitched/voiced segment. Compare each attack with the preceding 60 ms.
    frame = max(1, round(rate * .005))
    padded = np.pad(signal, (0, (-len(signal)) % frame))
    energy = np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1))
    baseline = np.convolve(np.pad(energy, (12, 0)), np.ones(12) / 12, mode="valid")[:len(energy)]
    novelty = np.maximum(energy - baseline, 0)
    floor = float(np.percentile(energy, 20))
    threshold = max(.00035, floor * 2.5, float(np.max(novelty)) * .035)
    candidates = [i for i in range(len(energy))
                  if novelty[i] > threshold and energy[i] > max(.0007, floor * 3)
                  and energy[i] > baseline[i] * 1.55
                  and novelty[i] >= np.max(novelty[max(0, i - 2):i + 3])]
    # Keep the strongest attack in each 75 ms neighbourhood, suppressing desk
    # resonance and microphone ringing while allowing quick repeated beats.
    selected = []
    separation = max(1, round(.075 * rate / frame))
    for index in sorted(candidates, key=lambda i: novelty[i], reverse=True):
        if all(abs(index - other) >= separation for other in selected):
            selected.append(index)
    if not selected:
        raise MelodyError("I couldn't hear distinct taps. Move the microphone closer and tap a clear rhythm.")
    maximum = max(float(energy[i]) for i in selected)
    return [Note(38, max(0., (i - .5) * frame / rate), .10,
                 min(120, max(40, round(100 * (float(energy[i]) / maximum) ** .5))))
            for i in sorted(selected)]


def analyze_recording(audio: np.ndarray, rate: int, mode="auto"):
    if mode == "percussion":
        return detect_taps(audio, rate), "percussion"
    if mode not in ("auto", "melody"):
        raise ValueError("Unknown recording type")
    try:
        return transcribe(audio, rate), "melody"
    except MelodyError:
        if mode == "melody":
            raise
        try:
            return detect_taps(audio, rate), "percussion"
        except MelodyError as error:
            raise MelodyError("I couldn't hear clear notes or taps. Move closer to the microphone and try again.") from error


def adjusted_notes(notes: list[Note], speed=1., octave=0) -> list[Note]:
    speed = min(1.75, max(.5, float(speed)))
    octave = min(2, max(-2, int(octave)))
    # Remove the wait before the first hum, retaining the played rhythm.
    first = min(n.start for n in notes)
    return [Note(min(127, max(0, n.pitch + octave * 12)),
                 (n.start - first) / speed + .12,
                 n.duration / speed, n.velocity) for n in notes]


class InstrumentPlayer:
    def __init__(self):
        self._search = os.add_dll_directory(str(ROOT / "Runtime" / "fluidsynth" / "bin"))
        self._library = C.CDLL(str(ROOT / "Runtime" / "fluidsynth" / "bin" / "libfluidsynth-3.dll"))
        self._lock = threading.Lock()
        specs = {
            "new_fluid_settings": (P, []), "delete_fluid_settings": (None, [P]),
            "fluid_settings_setnum": (I, [P, S, C.c_double]),
            "fluid_settings_setint": (I, [P, S, I]),
            "new_fluid_synth": (P, [P]), "delete_fluid_synth": (None, [P]),
            "fluid_synth_sfload": (I, [P, S, I]),
            "fluid_synth_program_select": (I, [P, I, U, U, U]),
            "fluid_synth_noteon": (I, [P, I, I, I]), "fluid_synth_noteoff": (I, [P, I, I]),
            "fluid_synth_all_sounds_off": (I, [P, I]),
            "fluid_synth_set_reverb_on": (None, [P, I]),
            "fluid_synth_set_chorus_on": (None, [P, I]),
            "fluid_synth_set_reverb": (I, [P, C.c_double, C.c_double, C.c_double, C.c_double]),
            "fluid_synth_write_float": (I, [P, I, P, I, I, P, I, I]),
        }
        for name, (result, arguments) in specs.items():
            function = getattr(self._library, name)
            function.restype, function.argtypes = result, arguments
        f = self._library
        self.settings = f.new_fluid_settings()
        f.fluid_settings_setnum(self.settings, b"synth.sample-rate", SAMPLE_RATE)
        f.fluid_settings_setnum(self.settings, b"synth.gain", .6)
        f.fluid_settings_setint(self.settings, b"synth.polyphony", 64)
        self.synth = f.new_fluid_synth(self.settings)
        if not self.synth:
            self.close()
            raise RuntimeError("The instrument player could not start")
        f.fluid_synth_set_chorus_on(self.synth, 0)
        self.font = f.fluid_synth_sfload(self.synth, os.fsencode(ROOT / "Runtime" / "MuseScore_General.sf2"), 1)
        if self.font < 0:
            self.close()
            raise RuntimeError("The installed instrument library could not be opened")

    def render(self, notes: list[Note], patch=0, speed=1., octave=0, room=.15,
               percussion=False, drum_note=38):
        if not notes:
            raise MelodyError("Record a melody first.")
        notes = adjusted_notes(notes, speed, 0 if percussion else octave)
        if percussion:
            notes = [Note(int(drum_note), n.start, n.duration, n.velocity) for n in notes]
        channel = 9 if percussion else 0
        with self._lock:
            f, synth = self._library, self.synth
            f.fluid_synth_all_sounds_off(synth, 0)
            f.fluid_synth_all_sounds_off(synth, 9)
            if f.fluid_synth_program_select(synth, channel, self.font, 128 if percussion else 0,
                                            0 if percussion else int(patch)) != 0:
                raise MelodyError("That instrument isn't available. Try another sound.")
            room = min(.7, max(0., float(room)))
            f.fluid_synth_set_reverb_on(synth, int(room > 0))
            f.fluid_synth_set_reverb(synth, .5, .35, .8, room)
            end = max(n.start + n.duration for n in notes) + (2.5 if room else 1.2)
            output = np.zeros((math.ceil(end * SAMPLE_RATE), 2), dtype=np.float32)
            events = []
            for note in notes:
                events += [(round(note.start * SAMPLE_RATE), 1, note),
                           (round((note.start + note.duration) * SAMPLE_RATE), 0, note)]
            previous = 0

            def fill(start, stop):
                while start < stop:
                    count = min(4096, stop - start)
                    destination = output[start:start + count]
                    ptr = destination.ctypes.data_as(P)
                    f.fluid_synth_write_float(synth, count, ptr, 0, 2, ptr, 1, 2)
                    start += count

            for frame, on, note in sorted(events, key=lambda event: (event[0], event[1])):
                fill(previous, frame)
                if on:
                    f.fluid_synth_noteon(synth, channel, note.pitch, note.velocity)
                else:
                    f.fluid_synth_noteoff(synth, channel, note.pitch)
                previous = frame
            fill(previous, len(output))
            if not np.isfinite(output).all():
                raise RuntimeError("The instrument produced invalid audio")
            peak = float(np.max(np.abs(output)))
            if peak < 1e-7:
                raise MelodyError("This instrument was silent. Please choose another sound.")
            output *= .78 / peak
            return output

    def close(self):
        if getattr(self, "synth", None):
            self._library.delete_fluid_synth(self.synth)
            self.synth = None
        if getattr(self, "settings", None):
            self._library.delete_fluid_settings(self.settings)
            self.settings = None
        if getattr(self, "_search", None):
            self._search.close()
            self._search = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def export_midi(path: Path, notes: list[Note], patch=0, speed=1., octave=0,
                percussion=False, drum_note=38):
    melody = mido.MidiFile(type=1, ticks_per_beat=960)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    melody.tracks.append(conductor)
    track = mido.MidiTrack()
    melody.tracks.append(track)
    channel = 9 if percussion else 0
    track.append(mido.Message("program_change", program=0 if percussion else int(patch), channel=channel, time=0))
    events = []
    for note in adjusted_notes(notes, speed, 0 if percussion else octave):
        if percussion:
            note = Note(int(drum_note), note.start, note.duration, note.velocity)
        events += [(round(note.start * 1920), 1, note),
                   (round((note.start + note.duration) * 1920), 0, note)]
    previous = 0
    for tick, on, note in sorted(events, key=lambda event: (event[0], event[1])):
        track.append(mido.Message("note_on" if on else "note_off", note=note.pitch,
                                  channel=channel, velocity=note.velocity if on else 0, time=tick - previous))
        previous = tick
    melody.save(str(path))


def run_job(job: dict):
    if job["kind"] == "analyze":
        audio, rate = read_audio(job["input"])
        notes, mode = analyze_recording(audio, rate, job.get("mode", "melody"))
        result = {"notes": [asdict(n) for n in notes], "input": job["input"],
                  "duration": len(audio) / rate, "mode": mode}
        write_json(Path(job["output"]), result)
        return result
    if job["kind"] == "render":
        saved = json.loads(Path(job["notes"]).read_text(encoding="utf-8"))
        notes = [Note(**n) for n in saved["notes"]]
        with InstrumentPlayer() as player:
            audio = player.render(notes, **job["settings"])
        sf.write(job["output"], audio, SAMPLE_RATE, subtype="PCM_16")
        return {"audio": job["output"], "duration": len(audio) / SAMPLE_RATE}
    raise ValueError("Unknown audio job")


if __name__ == "__main__":
    import argparse
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_job(json.loads(args.job.read_text(encoding="utf-8")))
        print("HM_RESULT:" + json.dumps({"ok": True, "result": result}), flush=True)
    except Exception as error:
        traceback.print_exc()
        print("HM_RESULT:" + json.dumps({"ok": False, "error": str(error),
              "friendly": isinstance(error, MelodyError)}), flush=True)
        raise SystemExit(1)
