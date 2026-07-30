from typing import Protocol, runtime_checkable

from app.domain.voice import AudioInput, AudioOutput, SpeechSynthesisOptions, Transcript


@runtime_checkable
class SpeechToTextService(Protocol):
    async def transcribe(self, audio: AudioInput) -> Transcript: ...


@runtime_checkable
class TextToSpeechService(Protocol):
    async def synthesize(
        self,
        text: str,
        options: SpeechSynthesisOptions,
    ) -> AudioOutput: ...
