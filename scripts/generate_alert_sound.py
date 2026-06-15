# scripts/generate_alert_sound.py
import wave, struct, math, os

os.makedirs("assets/sounds", exist_ok=True)

sample_rate = 44100
duration    = 0.4   # seconds
frequency   = 880   # Hz

n_samples = int(sample_rate * duration)
samples = [
    int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
    for i in range(n_samples)
]

with wave.open("assets/sounds/alert.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(struct.pack(f"<{n_samples}h", *samples))

print("assets/sounds/alert.wav created!")