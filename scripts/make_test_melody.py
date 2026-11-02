"""Create a known monophonic test signal with ordinary additive synthesis."""
from pathlib import Path
import json
import math
import struct
import wave

ROOT = Path(__file__).resolve().parents[1]
SR = 44100
notes = [(60, .6), (62, .6), (64, .9), (67, .6), (64, .6), (62, .6),
         (60, .9), (60, .6), (64, .6), (65, .6), (67, .9), (72, 1.2)]
samples = [0] * round(.5 * SR)
expected = []
for index, (pitch, duration) in enumerate(notes):
    start = len(samples) / SR
    hz = 440 * 2 ** ((pitch - 69) / 12)
    count = round(duration * SR)
    for frame in range(count):
        t = frame / SR
        envelope = min(1.0, t / .025, (duration - t) / .04)
        phase = 2 * math.pi * hz * t + .08 * math.sin(2 * math.pi * 4.8 * t)
        value = .28 * envelope * (math.sin(phase) + .22 * math.sin(2 * phase) + .08 * math.sin(3 * phase))
        samples.append(round(value * 32767))
    expected.append({"midi": pitch, "start_seconds": start, "duration_seconds": duration})
    samples.extend([0] * round((.3 if index == 6 else .16) * SR))
samples.extend([0] * SR)
output = ROOT / "examples" / "melody.wav"
output.parent.mkdir(exist_ok=True)
with wave.open(str(output), "wb") as target:
    target.setparams((1, 2, SR, 0, "NONE", "not compressed"))
    target.writeframes(struct.pack(f"<{len(samples)}h", *samples))
(ROOT / "examples" / "expected-notes.json").write_text(json.dumps({
    "description": "12-note, conventionally synthesised monophonic validation signal; not a human humming test",
    "sample_rate": SR, "duration_seconds": len(samples) / SR, "notes": expected,
}, indent=2), encoding="utf-8")
print(output)
