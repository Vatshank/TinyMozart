import asyncio
import os
import queue

from google import genai
from google.genai import types
import sounddevice as sd


API_KEY = os.getenv("GOOGLE_API_KEY", "PASTE_YOUR_GOOGLE_API_KEY_HERE")
SAMPLE_RATE = 48000
CHANNELS = 2
# NOTE: use the word "solo" for a single instrument (it might or might not listen)
# TEXT_PROMPT = "David Bowie's Let's Dance mixed with Metallica's Battery"
# TEXT_PROMPT = "introspective acoustic guitar. No other instruments. No vocals."
# TEXT_PROMPT = "Minimal techno"
# TEXT_PROMPT = "solo acoustic guitar"
# TEXT_PROMPT = "solo jazz guitar"
# TEXT_PROMPT = "jazz"
# TEXT_PROMPT = "Elliot smith style"
# TEXT_PROMPT = "Rage Against the Machine style"
# TEXT_PROMPT = "Elliot smith style acoustic"
TEXT_PROMPT = "Radiohead style drums and electric guitar"
# TEXT_PROMPT = ""

class StreamingAudioPlayer:
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._remainder = b""
        self._stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            callback=self._callback,
            blocksize=0,
        )

    def _callback(self, outdata, frames, _time, status):
        if status:
            # Keep running even if there are output underruns.
            pass

        requested_bytes = len(outdata)
        buffer = bytearray()

        if self._remainder:
            buffer.extend(self._remainder)
            self._remainder = b""

        while len(buffer) < requested_bytes:
            try:
                buffer.extend(self._queue.get_nowait())
            except queue.Empty:
                break

        if len(buffer) > requested_bytes:
            self._remainder = bytes(buffer[requested_bytes:])
            buffer = buffer[:requested_bytes]

        if len(buffer) < requested_bytes:
            buffer.extend(b"\x00" * (requested_bytes - len(buffer)))

        outdata[:] = bytes(buffer)

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def push(self, pcm_chunk: bytes):
        if pcm_chunk:
            self._queue.put(pcm_chunk)


def get_api_key() -> str:
    if API_KEY == "PASTE_YOUR_GOOGLE_API_KEY_HERE":
        raise ValueError(
            "Set GOOGLE_API_KEY or replace API_KEY placeholder in src/lyria_to_stems.py"
        )
    return API_KEY


async def main():
    client = genai.Client(api_key=get_api_key(), http_options={"api_version": "v1alpha"})
    player = StreamingAudioPlayer()

    async def receive_audio(session):
        """Example background task to process incoming audio."""
        async for message in session.receive():
            server_content = getattr(message, "server_content", None)
            if not server_content:
                continue

            audio_chunks = getattr(server_content, "audio_chunks", [])
            for chunk in audio_chunks:
                player.push(chunk.data)

            await asyncio.sleep(10**-12)

    try:
        player.start()
        async with (
            client.aio.live.music.connect(model="models/lyria-realtime-exp") as session,
            asyncio.TaskGroup() as tg,
        ):
            # Set up task to receive server messages.
            tg.create_task(receive_audio(session))

            # Send initial prompts and config
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text=TEXT_PROMPT, weight=1.0)]
            )
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(bpm=90, temperature=1.0,
                                                    #    mute_bass=True, mute_drums=True,
                                                    #    only_bass_and_drums=True
                                                       )
            )

            # Start streaming music
            await session.play()
            print("Streaming audio... Press Ctrl+C to stop.")
            await asyncio.Event().wait()
    finally:
        player.stop()


if __name__ == "__main__":
    asyncio.run(main())