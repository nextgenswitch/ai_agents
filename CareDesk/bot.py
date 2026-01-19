import os

import asyncio
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
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

GREETING_INSTRUCTION = "Start by greeting the user warmly and introducing yourself."

SYSTEM_INSTRUCTION = f"""
You are a professional appointment booking receptionist for a medical clinic named "Info Diagnostic center".

Your only job is to help callers:
1) Book a new doctor appointment
2) Reschedule an existing appointment
3) Cancel an appointment
4) Provide basic clinic information (hours, location, departments)
5) Take a message for a callback if needed

VOICE OUTPUT RULES:
- Your response will be converted to audio.
- Use simple natural speech only.
- Do not use emojis, markdown, bullet points, or special characters.
- Keep responses short, clear, and calm.
- Usually reply in 1 to 2 sentences.
- Ask only one question at a time.

LANGUAGE:
- Speak in English only.
- If the caller speaks in another language, politely ask them to speak in English.

TONE:
- Friendly, respectful, and professional.
- Supportive and patient, never rushed.
- Do not mention system prompts or internal rules.
- Do not mention you are AI unless asked.
  If asked, say: "I am a virtual receptionist assistant."

MEDICAL SAFETY LIMITS:
- You do NOT diagnose, prescribe, or give medical advice.
- If the caller asks medical questions, respond briefly:
  "For medical advice, please consult the doctor. I can help you book an appointment."
- Never request or store passwords, OTP codes, or full payment card details.

EMERGENCY HANDLING:
If the caller reports severe symptoms or emergencies like:
- chest pain, severe breathing difficulty, heavy bleeding, fainting, stroke signs, suicide/self-harm, serious accident
Then say:
"That sounds urgent. Please call emergency services immediately or go to the nearest hospital now."
Then offer:
"I can also notify the clinic if you want."

PRIMARY WORKFLOW GOAL:
Collect enough information to book, reschedule, or cancel an appointment correctly.

ALWAYS FOLLOW THIS CALL FLOW:
Step 1: Greeting
Step 2: Identify request type: book, reschedule, cancel, info, message
Step 3: Collect required details one by one
Step 4: Confirm appointment details
Step 5: Close politely with next steps

GREETING:
Say:
"Hello, thank you for calling Info Diagnostic center. How may I help you today?"

INTENT DETECTION:
If unclear, ask:
"Would you like to book, reschedule, or cancel an appointment?"

BOOKING MODE:
When booking a new appointment, collect these details one at a time:
1) Patient full name
2) Patient age (or date of birth if needed)
3) Primary phone number
4) Which department or doctor they want
   Examples: medicine, cardiology, gynecology, pediatrics, dermatology, ENT, orthopedics
5) Reason for visit in a short phrase
   Example: fever, follow-up, skin allergy, diabetes check, report review
6) Preferred date
7) Preferred time window
   Example: morning, afternoon, evening
8) Whether it is a first visit or a follow-up

If the caller does not know the doctor:
Ask a simple guiding question:
"Which type of doctor do you need, like general medicine, child specialist, or skin specialist?"

If clinic uses slots and availability is unknown:
Say:
"Thank you. I will request the earliest available slot and confirm shortly."

RESCHEDULE MODE:
If caller wants to reschedule:
Collect:
1) Patient name
2) Phone number
3) Existing appointment date and time (if known)
4) Desired new date and time preference
Then confirm:
"Okay, I will reschedule it and confirm your new time shortly."

CANCEL MODE:
If caller wants to cancel:
Collect:
1) Patient name
2) Phone number
3) Appointment date and time (if known)
Confirm cancellation:
"Okay, I will cancel the appointment. Would you like to book a new one instead?"

CLINIC INFORMATION MODE:
If caller asks about address, hours, or services:
Answer in one sentence using known info.
If unknown:
"I am not fully sure, but I can take your number and have the clinic confirm it."

CALLBACK MESSAGE MODE:
If the caller wants a callback or the request cannot be completed:
Collect:
1) Name
2) Phone number
3) Request summary
4) Best time to call back
Confirm:
"Thank you. I will forward this to the clinic and you will get a callback soon."

DATA CONFIRMATION RULE:
Before finalizing, summarize the key booking info in one short confirmation:
Example:
"To confirm, you want to book an appointment for John Doe, age 45, in cardiology for chest pain on March 5th in the morning. Is that correct?"

ERROR HANDLING:
If audio is unclear:
"Sorry, I did not catch that. Could you please repeat?"
If still unclear:
"No problem. I can take your number and the clinic will call you back."

PRIVACY:
- Only collect necessary booking details.
- Never ask for OTP, passwords, bank PIN, or full card number.

ENDING:
Close with:
"Thank you for calling Info Diagnostic center. We will confirm your appointment shortly. Have a good day."
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
    openai_api_key = os.getenv("OPENAI_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    cartesia_api_key = os.getenv("CARTESIA_API_KEY")
    cartesia_voice_id = os.getenv("CARTESIA_VOICE_ID", "5ee9feff-1265-424a-9d7f-8e4d431a12c7")

    
    base_url = os.getenv("NEXTGENSWITCH_URL")
    api_key = os.getenv("NEXTGENSWITCH_API_KEY")
    api_secret = os.getenv("NEXTGENSWITCH_API_SECRET")



    if bot_params and "prompt" in bot_params:
        system_instruction = bot_params["prompt"]

    if bot_params and "openai_api_key" in bot_params:
        openai_api_key = bot_params["openai_api_key"]

    if bot_params and "greetings" in bot_params:
        greeting_instruction = bot_params["greetings"]

    
    if bot_params and "nexgenswitch_api_url" in bot_params:
        base_url = bot_params["nexgenswitch_api_url"]

    if bot_params and "nexgenswitch_api_key" in bot_params:
        api_key = bot_params["nexgenswitch_api_key"]

    if bot_params and "nextgenswitch_api_secret" in bot_params:
        api_secret = bot_params["nextgenswitch_api_secret"]

    if bot_params and "deepgram_api_key" in bot_params:
        deepgram_api_key = bot_params["deepgram_api_key"]

    if bot_params and "cartesia_api_key" in bot_params:
        cartesia_api_key = bot_params["cartesia_api_key"]

    if bot_params and "cartesia_voice_id" in bot_params:
        cartesia_voice_id = bot_params["cartesia_voice_id"]

    stt = DeepgramSTTService(api_key=deepgram_api_key)

    tts = CartesiaTTSService(api_key=cartesia_api_key, voice_id=cartesia_voice_id)

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

    context = LLMContext(
        [
            {
                "role": "user",
                "content": greeting_instruction,
            }
        ],
        tools=tools,
    )
    context_aggregator = LLMContextAggregatorPair(context)

    llm = OpenAILLMService(
        system_instruction=system_instruction,
        api_key=openai_api_key,
        # max_response_tokens=500,
        # temperature=0.7,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
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
