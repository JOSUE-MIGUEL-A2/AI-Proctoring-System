# src/utils/audio_alert.py

import threading
import sys
import time


class AudioAlert:
    """
    Cross-platform beep that fires in a background thread so it never blocks
    the video pipeline.
    """

    def __init__(self, cooldown_sec: float = 3.0):
        self._cooldown = cooldown_sec
        self._last_played = 0.0
        self._lock = threading.Lock()

    def trigger(self):
        """Call this on each violation frame. Respects cooldown to avoid spam."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_played < self._cooldown:
                return
            self._last_played = now

        thread = threading.Thread(target=self._play, daemon=True)
        thread.start()

    def _play(self):
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(880, 400)   # 880 Hz, 400 ms
            else:
                # macOS / Linux — uses playsound with a bundled WAV if available
                try:
                    from playsound import playsound
                    playsound("assets/sounds/alert.wav", block=True)
                except Exception:
                    # Fallback: print the terminal bell character
                    print("\a", end="", flush=True)
        except Exception as e:
            print(f"[AudioAlert] Could not play sound: {e}")