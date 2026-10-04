"""Hands-free voice input: microphone -> text, fully offline.

A background thread records from the default microphone, cuts the stream into
utterances by loudness (speech starts when the level rises well above the
room noise, ends after a short silence), and transcribes each one with
faster-whisper. Finished text is collected with poll().

    python -m roarm_rl.voice          # print what it hears, Ctrl+C to stop

Needs `sounddevice` and `faster-whisper`. The first run downloads the speech
model (about 150 MB); after that nothing leaves the machine.
"""

import queue
import threading
import time

import numpy as np

RATE = 16000
BLOCK = 480  # 30 ms
MODEL = "base.en"

START_FACTOR = 3.5  # speech starts at this multiple of the noise floor...
MIN_START_RMS = 0.008  # ...but never below this level
END_SILENCE_S = 0.7
MIN_SPEECH_S = 0.35
MAX_SPEECH_S = 12.0
PRE_ROLL_S = 0.3  # audio kept from just before speech was detected

# Biases recognition toward the words the arm understands ("nod", not "not").
HOTWORDS = ("nod wave bow shake shrug point look clap grab dance twist wiggle salute beckon "
            "peek sway stir zigzag high five fist bump handshake gripper slowly twice")


class Segmenter:
    """Cuts a stream of audio blocks into utterances by loudness."""

    def __init__(self):
        self.noise = 0.004
        self.level = 0.0
        self._pre = []
        self._speech = []
        self._silence = 0.0

    @property
    def speaking(self):
        return bool(self._speech)

    def push(self, block):
        """Feed one block of float32 samples; returns a finished utterance or None."""
        rms = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
        self.level = rms
        seconds = len(block) / RATE
        threshold = max(MIN_START_RMS, self.noise * START_FACTOR)

        if not self._speech:
            if rms < threshold:
                self.noise = 0.95 * self.noise + 0.05 * rms
                self._pre.append(block)
                if len(self._pre) * seconds > PRE_ROLL_S:
                    self._pre.pop(0)
                return None
            self._speech = self._pre + [block]
            self._pre = []
            self._silence = 0.0
            return None

        self._speech.append(block)
        self._silence = 0.0 if rms >= threshold * 0.6 else self._silence + seconds
        length = sum(len(b) for b in self._speech) / RATE
        if self._silence >= END_SILENCE_S or length >= MAX_SPEECH_S:
            audio = np.concatenate(self._speech)
            self._speech = []
            if length - self._silence >= MIN_SPEECH_S:
                return audio
        return None


class Transcriber:
    def __init__(self, model=MODEL):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, audio):
        """Text for one utterance, or '' when it does not look like speech."""
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=3, hotwords=HOTWORDS,
            condition_on_previous_text=False, without_timestamps=True,
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 or seg.avg_logprob < -1.2:
                continue
            parts.append(seg.text.strip())
        return " ".join(parts).strip()


class VoiceListener:
    """Listens on a background thread. status: off, loading, listening, hearing, thinking, error."""

    def __init__(self):
        self.status = "off"
        self.error = None
        self.enabled = True  # False = muted: audio is ignored
        self._texts = queue.Queue()
        self._audio = queue.Queue()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.status = "loading"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def poll(self):
        """Utterances transcribed since the last call."""
        out = []
        while True:
            try:
                out.append(self._texts.get_nowait())
            except queue.Empty:
                return out

    def _open_stream(self, sd):
        """Open the default mic, at 16 kHz if it allows, else at its own rate."""
        device = sd.query_devices(kind="input")
        native = int(device["default_samplerate"])
        channels = min(2, int(device["max_input_channels"]))
        errors = []
        for rate, ch in ((RATE, 1), (native, 1), (native, channels)):
            def on_audio(indata, frames, time_info, status, rate=rate):
                mono = indata.mean(axis=1)
                if rate != RATE:  # linear resample; plenty for speech
                    n = int(round(len(mono) * RATE / rate))
                    mono = np.interp(np.linspace(0, len(mono), n, endpoint=False),
                                     np.arange(len(mono)), mono)
                self._audio.put(mono.astype(np.float32))

            try:
                stream = sd.InputStream(samplerate=rate, channels=ch, dtype="float32",
                                        blocksize=int(BLOCK * rate / RATE), callback=on_audio)
                stream.start()
                return stream
            except Exception as e:
                errors.append(str(e))
        raise RuntimeError(
            "could not open the microphone (check Windows Settings > Privacy & security > "
            f"Microphone, and that no other app holds it): {errors[0]}")

    def _run(self):
        try:
            import sounddevice as sd

            stream = self._open_stream(sd)  # fail fast, before the slow model load
            transcriber = Transcriber()
            segmenter = Segmenter()
            with stream:
                while not self._audio.empty():
                    self._audio.get_nowait()
                self.status = "listening"
                while not self._stop.is_set():
                    try:
                        block = self._audio.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if not self.enabled:
                        segmenter = Segmenter()
                        self.status = "muted"
                        continue
                    utterance = segmenter.push(block)
                    self.status = "hearing" if segmenter.speaking else "listening"
                    if utterance is not None:
                        self.status = "thinking"
                        text = transcriber.transcribe(utterance)
                        if text:
                            self._texts.put(text)
                        # drop what piled up while transcribing (often the arm's own noise)
                        while not self._audio.empty():
                            self._audio.get_nowait()
                        self.status = "listening"
        except Exception as e:
            self.error = str(e)
            self.status = "error"


def main():
    listener = VoiceListener()
    listener.start()
    print("loading the speech model...")
    last = None
    try:
        while listener.status != "error":
            if listener.status != last:
                last = listener.status
                print(f"[{last}]")
            for text in listener.poll():
                print("heard:", text)
            time.sleep(0.05)
        print("voice input failed:", listener.error)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
