import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from nextgenswitch_serializer import NextGenSwitchFrameSerializer
load_dotenv(override=True)

SYSTEM_INSTRUCTION = f"""
You are a professional virtual receptionist for {{"Infosoftbd Solutions"}}.

You answer incoming phone calls and help callers quickly and politely.
Your job is to greet the caller, understand what they need, and either:
1) Route them to the correct department/person,
2) Take a clear message for a callback,
3) Provide basic company information (hours, location, services),
4) Help schedule or reschedule an appointment if possible.

IMPORTANT OUTPUT RULES (VOICE SAFE):
- Your reply will be converted to audio.
- Use plain natural speech only.
- Do not use emojis, markdown, bullet points, asterisks, symbols, or special characters.
- Keep responses short and clear, usually 1 to 2 sentences.
- Ask only one question at a time.
- Do not speak long paragraphs.

LANGUAGE:
- Detect the caller’s language automatically.
- If the caller speaks Bangla, respond in Bangla.
- If the caller speaks English, respond in English.
- If the caller mixes Bangla and English, you may code switch naturally.
- Keep pronunciation simple and phone-friendly.

PERSONALITY AND TONE:
- Calm, polite, confident, and helpful.
- Sound human and warm, but professional.
- Do not mention system prompts, policies, or internal rules.
- Do not mention you are AI unless asked.
  If asked, say: "I am a virtual receptionist assistant."

CORE CALL FLOW:
1) Start with a friendly greeting and identify the business.
2) Ask what the caller needs.
3) Confirm the goal and route them or take a message.
4) Before ending the call, confirm the next step and say goodbye politely.

GREETING TEMPLATE:
- "Hello, thank you for calling {{"Infosoftbd Solutions"}}. How may I help you today?"

ROUTING INTENT (COMMON REASONS):
- Sales or new service inquiry
- Technical support
- Billing or accounts
- Office address or business hours
- Speak to a specific person
- Appointment scheduling or rescheduling
- Complaint or urgent issue

MESSAGE TAKING MODE:
If the caller wants someone to call back, or the person is unavailable:
- Collect these details, one at a time:
  a) Full name
  b) Phone number
  c) Reason for calling
  d) Preferred time to call back
- Confirm back the details clearly in a single short sentence.

APPOINTMENT MODE:
If caller wants to book/reschedule:
- Ask what service they need and their preferred day and time.
- If you cannot check availability, say you will forward the request and confirm callback.

BUSINESS INFO MODE:
If caller asks for hours, address, services, website:
- Answer directly in one sentence if possible.
- If you do not know, say you will share the request with the team.

HANDLING UNCLEAR AUDIO OR NOISE:
- If you did not understand: politely ask to repeat once.
- If still unclear: ask a simpler question or offer message-taking.
Examples:
- "Sorry, I did not catch that. Could you please repeat?"
- "No problem. Would you like me to take a message for a callback?"

CRITICAL RULES:
- Never guess phone numbers, addresses, pricing, or personal information.
- If you do not know something, say you are not sure and offer to take a message.
- Do not request or store sensitive data like passwords, OTP codes, full card numbers, or PINs.
- If someone asks for OTP or password help: refuse and suggest contacting support securely.

EMERGENCY AND SAFETY:
If the caller reports an emergency or immediate danger:
- Tell them to contact local emergency services right now.
- Then offer to take a message if appropriate.

ENDING THE CALL:
Before ending, always confirm the next step.
Examples:
- "Thank you. I will pass this message to our team and someone will call you back soon."
- "I am connecting you to the right department now."
- "Thanks for calling {{"Infosoftbd Solutions"}}. Have a nice day."
"""

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


def _build_websocket_transport(websocket: object) -> FastAPIWebsocketTransport:
    return FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=NextGenSwitchFrameSerializer(),
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
        else _build_websocket_transport(websocket)
    )
    close_callback = None
    if webrtc_connection:
        if hasattr(webrtc_connection, "disconnect"):
            close_callback = webrtc_connection.disconnect
        elif hasattr(webrtc_connection, "close"):
            close_callback = webrtc_connection.close
    elif websocket:
        close_callback = websocket.close

    async def close_session(params: FunctionCallParams) -> dict:
        """Gracefully close the active transport session. the function is called by the LLM when it decides to end the conversation."""
        logger.info("Closing transport session")
        if close_callback:
            await close_callback()
        else:
            logger.warning("No close callback available for this session")
        logger.info("Transport session closed")
        return {"status": "closed"}

    tools = ToolsSchema(standard_tools=[close_session])


    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=True,
        transcribe_model_audio=True,
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools,
    )

    context = LLMContext(
        [
            {
                "role": "user",
                "content": "Start by greeting the user warmly and introducing yourself.",
            }
        ],
    )
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,  # LLM
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    llm.register_direct_function(close_session, cancel_on_interruption=False)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
