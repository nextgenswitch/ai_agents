import os
import re
import json
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    DataFrame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    TTSSpeakFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from nextgenswitch_serializer import NextGenSwitchFrameSerializer
from tools.support_ticket import create_ticket
from tools.transfer_call import transfer_call
from llm_service import get_llm_service
from stt_service import get_stt_service
from tts_service import get_tts_service

load_dotenv(override=True)

GREETING_INSTRUCTION = "Start by greeting the user warmly and introducing yourself."

CLOSING_ANNOUNCEMENT = "Thank you for calling. This session will now close. Goodbye."

SYSTEM_INSTRUCTION = """You are a helpful AI voice assistant. Engage in natural and friendly conversations with users. Provide information, answer questions, and assist with tasks as needed. Always maintain a polite and professional tone.

CRITICAL RULES:
Your output will be converted to speech (text-to-speech). You MUST respond in plain text only. NEVER use:
   - Asterisks for bold or emphasis
   - Underscores for italic
   - Hashtags for headers
   - Backticks for code
   - Bullet points or numbered lists with symbols
   - Emojis or special characters
   - Any markdown or formatting whatsoever

Write naturally as if you're speaking out loud."""


class TranscriptProcessor(FrameProcessor):
    """Processor that captures transcripts and sends them via callback."""
    
    def __init__(self, get_callback, role: str):
        super().__init__()
        self._get_callback = get_callback  # Function that returns current callback
        self._role = role
        self._bot_text_buffer = ""
    
    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        callback = self._get_callback()  # Get current callback
        
        if isinstance(frame, TranscriptionFrame):
            # Final user transcription
            if callback:
                logger.debug(f"User final transcript: {frame.text}")
                await callback({
                    "type": "transcript",
                    "role": "user",
                    "text": frame.text,
                    "final": True
                })
        elif isinstance(frame, InterimTranscriptionFrame):
            # Interim user transcription
            if callback:
                logger.debug(f"User interim transcript: {frame.text}")
                await callback({
                    "type": "transcript",
                    "role": "user",
                    "text": frame.text,
                    "final": False
                })
        elif isinstance(frame, TextFrame) and self._role == "bot":
            # Bot text output (LLM response)
            if callback and frame.text:
                self._bot_text_buffer += frame.text
                logger.debug(f"Bot transcript: {frame.text}")
                await callback({
                    "type": "transcript",
                    "role": "bot",
                    "text": frame.text,
                    "final": False
                })
        
        await self.push_frame(frame, direction)


@dataclass
class CloseSessionFrame(DataFrame):
    """Signals the transport to close after queued audio has finished."""


class CloseSessionProcessor(FrameProcessor):
    def __init__(self, close_callback):
        super().__init__()
        self._close_callback = close_callback
        self._closing = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, CloseSessionFrame):
            if self._closing:
                return
            self._closing = True
            logger.info("Closing transport session after announcement")
            if self._close_callback:
                await self._close_callback()
            else:
                logger.warning("No close callback available for this session")
            return

        await self.push_frame(frame, direction)




def _build_webrtc_transport(webrtc_connection: object) -> SmallWebRTCTransport:
    return SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )


def _build_websocket_transport(
    websocket: object, stream_id: str | None = None
) -> FastAPIWebsocketTransport:
    serializer = NextGenSwitchFrameSerializer()
    if stream_id:
        serializer.set_stream_id(stream_id)
    return FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )


async def run_bot(
    webrtc_connection=None,
    websocket=None,
    stream_id=None,
    call_sid=None,
    bot_params=None,
):
    if webrtc_connection and websocket:
        raise ValueError("Provide either webrtc_connection or websocket, not both.")
    if not webrtc_connection and not websocket:
        raise ValueError("No transport input provided to run_bot.")

    transport = (
        _build_webrtc_transport(webrtc_connection)
        if webrtc_connection
        else _build_websocket_transport(websocket, stream_id=stream_id)
    )
    close_callback = None
    if webrtc_connection:
        if hasattr(webrtc_connection, "disconnect"):
            close_callback = webrtc_connection.disconnect
        elif hasattr(webrtc_connection, "close"):
            close_callback = webrtc_connection.close
    elif websocket:
        close_callback = websocket.close
    
    


    system_instruction = SYSTEM_INSTRUCTION
    greeting_instruction = GREETING_INSTRUCTION
    closing_announcement = os.getenv("CLOSING_ANNOUNCEMENT", CLOSING_ANNOUNCEMENT)
    default_forwarding_number = os.getenv("FORWARDING_NUMBER")
    base_url = os.getenv("NEXTGENSWITCH_URL")
    api_key = os.getenv("NEXTGENSWITCH_API_KEY")
    api_secret = os.getenv("NEXTGENSWITCH_API_SECRET")

    # Load from bot_params (agent config from JSON file)
    if bot_params:
        if "prompt" in bot_params and bot_params["prompt"]:
            system_instruction = bot_params["prompt"]
        if "greetings" in bot_params and bot_params["greetings"]:
            greeting_instruction = bot_params["greetings"]
        if "closing_announcement" in bot_params:
            closing_announcement = bot_params["closing_announcement"]
        if "forwarding_number" in bot_params:
            default_forwarding_number = bot_params["forwarding_number"]
        if "nexgenswitch_api_url" in bot_params:
            base_url = bot_params["nexgenswitch_api_url"]
        if "nexgenswitch_api_key" in bot_params:
            api_key = bot_params["nexgenswitch_api_key"]
        if "nextgenswitch_api_secret" in bot_params:
            api_secret = bot_params["nextgenswitch_api_secret"]
    
    logger.info(f"Bot starting with agent: {bot_params.get('agent') if bot_params else 'default'}")

    async def close_session(params: FunctionCallParams) -> dict:
        """Gracefully close the active transport session. the function is called by the LLM when it decides to end the conversation."""
        logger.info("Close session requested")
        if not close_callback:
            logger.warning("No close callback available for this session")
            return {"status": "closed"}
        if not tts:
            await close_callback()
            logger.info("Transport session closed")
            return {"status": "closed"}
        announcement = (closing_announcement or "").strip()
        if announcement:
            await tts.queue_frame(TTSSpeakFrame(announcement))
        await tts.queue_frame(CloseSessionFrame())
        return {"status": "closing"}

    async def transfer_call_to(params: FunctionCallParams, forwarding_number: int) -> None:
        """Trasfer call to agent. Call this function immidiately if user want to talk to a specific agent.

        This is a placeholder function to demonstrate how to transfer the call
        into a specific agent.
        """
        if not forwarding_number:
            logger.error("Forwarding number is not configured; skipping transfer for {}", call_sid)
            await params.result_callback({"status": "unsupported"})
            return
        
        if not base_url:
            logger.error("NEXTGENSWITCH_URL is not configured; unable to transfer call {}", call_sid)
            await params.result_callback({"status": "unsupported"})
            return
        if not api_key:
            logger.error("NEXTGENSWITCH_API_KEY is not configured; unable to transfer call {}", call_sid)
            await params.result_callback({"status": "unsupported"})
            return
        if not api_secret:
            logger.error("NEXTGENSWITCH_API_SECRET is not configured; unable to transfer call {}", call_sid)
            await params.result_callback({"status": "unsupported"})
            return
        

        logger.info("Transferring call {} to  {}", call_sid, forwarding_number)
        asyncio.create_task(transfer_call(call_sid, forwarding_number, base_url, api_key, api_secret))
        await params.result_callback({"status": "transferred"})

    async def support_ticket(
        params: FunctionCallParams,
        subject: str,
        description: str,
        name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
    ) -> None:
        """Create a support ticket for the current call."""
        if not call_sid:
            logger.error("Support ticket requested without call SID")
            await params.result_callback({"status": "unsupported"})
            return

        success = await create_ticket(
            call_sid=call_sid,
            subject=subject,
            description=description,
            name=name,
            email=email,
            phone=phone,
            base_url=base_url,
            api_key=api_key,
            api_secret=api_secret,
        )
        await params.result_callback({"status": "created" if success else "failed"})

    tools = ToolsSchema(standard_tools=[close_session, transfer_call_to, support_ticket])

    effective_params = bot_params or {}
    stt = get_stt_service(effective_params)
    tts = get_tts_service(effective_params)
    llm = get_llm_service(effective_params)

    # Set up transcript callback - will be configured after transport is ready
    transcript_callback = None
    
    def get_transcript_callback():
        return transcript_callback
    
    async def send_transcript(data):
        nonlocal transcript_callback
        if transcript_callback:
            try:
                await transcript_callback(data)
            except Exception as e:
                logger.debug(f"Transcript send error: {e}")

    # Create transcript processors with getter function
    user_transcript_processor = TranscriptProcessor(get_transcript_callback, "user")
    bot_transcript_processor = TranscriptProcessor(get_transcript_callback, "bot")
    # markdown_stripper = MarkdownStripper()

    context = LLMContext(
        [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": greeting_instruction},
        ],
        tools=tools,
    )
    context_aggregator = LLMContextAggregatorPair(context)
    close_session_processor = CloseSessionProcessor(close_callback)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_transcript_processor,
            context_aggregator.user(),
            llm,
            # markdown_stripper,
            bot_transcript_processor,
            tts,
            transport.output(),
            context_aggregator.assistant(),
            close_session_processor,
        ]
    )
    llm.register_direct_function(close_session, cancel_on_interruption=False)
    llm.register_direct_function(transfer_call_to, cancel_on_interruption=False)
    llm.register_direct_function(support_ticket, cancel_on_interruption=False)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal transcript_callback
        logger.info("Pipecat Client connected")
        
        # Set up the transcript callback using webrtc_connection's send_app_message
        # The 'client' parameter IS the webrtc_connection
        def _send_via_connection(data):
            try:
                client.send_app_message(data)  # Already handles JSON serialization
                logger.debug(f"Sent transcript: {data}")
            except Exception as e:
                logger.warning(f"Failed to send transcript: {e}")
        
        # Wrap in async for compatibility with TranscriptProcessor
        async def _async_send(data):
            _send_via_connection(data)
        
        transcript_callback = _async_send
        
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
