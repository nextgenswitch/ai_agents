# nextgenswitch_serializer.py
#
# NextGenSwitchFrameSerializer
# Twilio-like JSON WS:
#   {"event":"start","streamId":"..."}
#   {"event":"media","streamId":"...","media":{"payload":"<base64 ulaw 8k>"}}
#   {"event":"stop","streamId":"..."}
#
# This serializer:
# - Decodes inbound base64 μ-law @ wire_sample_rate (default 8000)
# - Converts to PCM16 and resamples to pipeline input sample_rate (e.g., 16000)
# - Encodes outbound PCM to μ-law @ wire_sample_rate and wraps in Twilio-like media JSON
#
# It intentionally mirrors Pipecat’s TwilioFrameSerializer conversion approach. :contentReference[oaicite:1]{index=1}

from __future__ import annotations

import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import Frame, StartFrame, AudioRawFrame, InputAudioRawFrame, TTSAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer


class NextGenSwitchSerializerParams(BaseModel):
    wire_sample_rate: int = 8000       # Twilio-style μ-law
    stt_sample_rate: int = 16000       # desired pipeline input rate (override if needed)


class NextGenSwitchFrameSerializer(FrameSerializer):
    """
    Serializer compatible with FastAPIWebsocketTransport.

    Pipecat contract:
      - setup(StartFrame)
      - serialize(Frame) -> str|bytes|None
      - deserialize(str|bytes) -> Frame|None
    """

    def __init__(self, params: Optional[NextGenSwitchSerializerParams] = None):
        self._params = params or NextGenSwitchSerializerParams()

        self._stream_id: Optional[str] = None

        self._wire_sr = int(self._params.wire_sample_rate)
        self._pipeline_in_sr = 0  # set in setup()
        self._pipeline_out_sr = 0  # optional; usually not needed for ulaw conversion

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

        self._dbg_in_count = 0
        self._dbg_out_count = 0

    def set_stream_id(self, stream_id: str) -> None:
        self._stream_id = stream_id

    async def setup(self, frame: StartFrame):
        # Pipecat passes pipeline configuration in StartFrame. Twilio serializer uses audio_in_sample_rate. :contentReference[oaicite:2]{index=2}
        self._pipeline_in_sr = int(frame.audio_in_sample_rate or self._params.stt_sample_rate)
        self._pipeline_out_sr = int(frame.audio_out_sample_rate or 0)

        logger.info(
            "Serializer setup: "
            f"audio_in_sample_rate={frame.audio_in_sample_rate}, "
            f"audio_out_sample_rate={frame.audio_out_sample_rate}, "
            f"wire_sr={self._wire_sr}, pipeline_in_sr={self._pipeline_in_sr}"
        )

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """
        Inbound: Twilio-like 'media' with base64 μ-law payload.
        Convert μ-law->PCM16 and resample 8k->pipeline_in_sr.
        """
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="ignore")
            msg = json.loads(data)
        except Exception:
            return None

        event = msg.get("event")
        if event != "media":
            # ignore start/stop/connected/etc.
            return None

        media = msg.get("media") or {}
        payload_b64 = media.get("payload")
        if not payload_b64:
            return None

        try:
            ulaw_bytes = base64.b64decode(payload_b64)
        except Exception:
            return None

        # Convert 8k μ-law -> PCM at pipeline input sample rate (same as Pipecat Twilio serializer). :contentReference[oaicite:3]{index=3}
        pcm = await ulaw_to_pcm(ulaw_bytes, self._wire_sr, self._pipeline_in_sr, self._input_resampler)
        if not pcm:
            return None

        # Optional lightweight debug every ~50 frames
        self._dbg_in_count += 1
        if self._dbg_in_count % 50 == 0:
            logger.debug(f"[SER IN] ulaw={len(ulaw_bytes)} bytes -> pcm={len(pcm)} bytes @ {self._pipeline_in_sr}")

        return InputAudioRawFrame(audio=pcm, num_channels=1, sample_rate=self._pipeline_in_sr)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """
        Outbound: Convert PCM at frame.sample_rate -> 8k μ-law and wrap as Twilio-like media JSON.
        """
        if not isinstance(frame, (AudioRawFrame, TTSAudioRawFrame)):
            return None

        if not self._stream_id:
            # If you want hard-fail here, raise. For safety, just drop.
            logger.warning("serialize(): missing stream_id; dropping outbound audio")
            return None

        pcm = frame.audio
        if not pcm:
            return None

        ulaw_bytes = await pcm_to_ulaw(pcm, frame.sample_rate, self._wire_sr, self._output_resampler)
        if not ulaw_bytes:
            return None

        payload = base64.b64encode(ulaw_bytes).decode("utf-8")

        self._dbg_out_count += 1
        if self._dbg_out_count % 50 == 0:
            logger.debug(
                f"[SER OUT] pcm={len(pcm)} bytes @ {frame.sample_rate} -> ulaw={len(ulaw_bytes)} bytes @ {self._wire_sr}"
            )

        answer = {
            "event": "media",
            "streamId": self._stream_id,  # your protocol uses streamId
            "media": {"payload": payload},
        }
        return json.dumps(answer)
