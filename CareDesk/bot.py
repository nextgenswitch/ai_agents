import os
import re
import sys
import threading
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from openpyxl import Workbook, load_workbook

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
from pipecat.services.llm_service import FunctionCallParams

from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from pipecat.adapters.schemas.tools_schema import ToolsSchema

from nextgenswitch_serializer import NextGenSwitchFrameSerializer


load_dotenv(override=True)

logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))


GREETING_INSTRUCTION = "Start the call now with the required greeting."

APPOINTMENTS_HEADERS = [
    "Logged At",
    "Action",
    "Patient Name",
    "Patient Age or DOB",
    "Phone",
    "Department or Doctor",
    "Reason",
    "Preferred Date",
    "Preferred Time",
    "Visit Type",
    "Existing Appointment",
    "Notes",
]

APPOINTMENT_LOG_LOCK = threading.Lock()

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
If caller wants to reschedule, collect:
1) Patient name
2) Phone number
3) Existing appointment date and time (if known)
4) Desired new date and time preference
Then confirm:
"Okay, I will reschedule it and confirm your new time shortly."

CANCEL MODE:
If caller wants to cancel, collect:
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
If the caller wants a callback or the request cannot be completed, collect:
1) Name
2) Phone number
3) Request summary
4) Best time to call back
Confirm:
"Thank you. I will forward this to the clinic and you will get a callback soon."

DATA CONFIRMATION RULE:
Before finalizing, summarize the key booking info in one short confirmation:
"To confirm, you want to book an appointment for John Doe, age 45, in cardiology for chest pain on March 5th in the morning. Is that correct?"

APPOINTMENT LOGGING TOOL:
After confirming a booking, reschedule, or cancellation, call the log_appointment tool once.
Use these fields: action, patient_name, patient_age_or_dob, phone, department_or_doctor, reason, preferred_date,
preferred_time, visit_type, existing_appointment, notes.
For reschedule or cancel, include existing_appointment if known.
Use YYYY-MM-DD for dates when possible.

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
""".strip()


def _resolve_appointments_path() -> str:
    env_path = os.getenv("APPOINTMENTS_XLSX_PATH")
    if env_path:
        return os.path.abspath(env_path)
    return os.path.join(os.path.dirname(__file__), "appointments.xlsx")


def _normalize_date(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return cleaned


def _safe_sheet_name(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[\\/*?:\\[\\]]", "-", value).strip()
    if not cleaned:
        return ""
    return cleaned[:31]


def _append_row_to_workbook(path: str, sheet_name: str, row: list[str]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(path):
        workbook = load_workbook(path)
        if sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
        else:
            sheet = workbook.create_sheet(sheet_name)
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = sheet_name

    if sheet.max_row == 1 and sheet.max_column == 1 and sheet["A1"].value is None:
        sheet.append(APPOINTMENTS_HEADERS)

    sheet.append(row)
    workbook.save(path)


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
    websocket: object,
    stream_id: str | None = None,
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
    """
    A Pipecat bot runner that supports either:
    - WebRTC transport (SmallWebRTCTransport)
    - FastAPI websocket transport (FastAPIWebsocketTransport)
    """

    if webrtc_connection and websocket:
        raise ValueError("Provide either webrtc_connection or websocket, not both.")
    if not webrtc_connection and not websocket:
        raise ValueError("No transport input provided to run_bot.")

    # Pick transport
    transport = (
        _build_webrtc_transport(webrtc_connection)
        if webrtc_connection
        else _build_websocket_transport(websocket, stream_id=stream_id)
    )

    # Determine close callback
    close_callback = None
    if webrtc_connection:
        if hasattr(webrtc_connection, "disconnect"):
            close_callback = webrtc_connection.disconnect
        elif hasattr(webrtc_connection, "close"):
            close_callback = webrtc_connection.close
    elif websocket:
        close_callback = websocket.close

    # --------------------------------------------------------
    # Load params (env defaults, override by bot_params)
    # --------------------------------------------------------
    system_instruction = SYSTEM_INSTRUCTION
    greeting_instruction = GREETING_INSTRUCTION

    openai_api_key = os.getenv("OPENAI_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

    cartesia_api_key = os.getenv("CARTESIA_API_KEY")
    cartesia_voice_id = os.getenv(
        "CARTESIA_VOICE_ID", "5ee9feff-1265-424a-9d7f-8e4d431a12c7"
    )

    # NextGenSwitch config (kept for your external usage)
    base_url = os.getenv("NEXTGENSWITCH_URL")
    api_key = os.getenv("NEXTGENSWITCH_API_KEY")
    api_secret = os.getenv("NEXTGENSWITCH_API_SECRET")

    # Overrides from bot_params
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

    # Basic validation to prevent silent failures
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY (env or bot_params.openai_api_key)")
    if not deepgram_api_key:
        raise ValueError("Missing DEEPGRAM_API_KEY (env or bot_params.deepgram_api_key)")
    if not cartesia_api_key:
        raise ValueError("Missing CARTESIA_API_KEY (env or bot_params.cartesia_api_key)")


    stt = DeepgramSTTService(api_key=deepgram_api_key)
    tts = CartesiaTTSService(api_key=cartesia_api_key, voice_id=cartesia_voice_id)


    async def close_session(params: FunctionCallParams) -> dict:
        """
        Gracefully close the active transport session.
        This function is called by the LLM when it decides to end the conversation.
        """
        logger.info("Closing transport session")
        if close_callback:
            await close_callback()
        else:
            logger.warning("No close callback available for this session")
        logger.info("Transport session closed")
        return {"status": "closed"}

    async def log_appointment(
        params: FunctionCallParams,
        action: str,
        patient_name: str | None = None,
        patient_age_or_dob: str | None = None,
        phone: str | None = None,
        department_or_doctor: str | None = None,
        reason: str | None = None,
        preferred_date: str | None = None,
        preferred_time: str | None = None,
        visit_type: str | None = None,
        existing_appointment: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Log appointment details to the Excel workbook.

        Args:
            action: Appointment action such as book, reschedule, or cancel.
            patient_name: Patient full name.
            patient_age_or_dob: Patient age or date of birth.
            phone: Primary phone number.
            department_or_doctor: Requested department or doctor.
            reason: Reason for the visit.
            preferred_date: Preferred appointment date.
            preferred_time: Preferred time window.
            visit_type: First visit or follow-up.
            existing_appointment: Existing appointment date/time (for reschedule or cancel).
            notes: Extra details or message.
        """
        target_date = _normalize_date(preferred_date or existing_appointment)
        if not target_date:
            target_date = datetime.now().strftime("%Y-%m-%d")

        sheet_name = _safe_sheet_name(target_date) or datetime.now().strftime("%Y-%m-%d")
        log_path = _resolve_appointments_path()
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            action,
            patient_name or "",
            patient_age_or_dob or "",
            phone or "",
            department_or_doctor or "",
            reason or "",
            preferred_date or "",
            preferred_time or "",
            visit_type or "",
            existing_appointment or "",
            notes or "",
        ]

        with APPOINTMENT_LOG_LOCK:
            _append_row_to_workbook(log_path, sheet_name, row)

        await params.result_callback(
            {"status": "logged", "path": log_path, "sheet": sheet_name}
        )

    tools = ToolsSchema(standard_tools=[close_session, log_appointment])

    context = LLMContext(
        [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": greeting_instruction},
        ],
        tools=tools,
    )
    context_aggregator = LLMContextAggregatorPair(context)

    llm = OpenAILLMService(
        api_key=openai_api_key,
        # model="gpt-4.1-mini",   # optional (depends on your pipecat version)
        # temperature=0.4,        # optional
        # max_response_tokens=250 # optional
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

    # Allow direct tool call
    llm.register_direct_function(close_session, cancel_on_interruption=False)
    llm.register_direct_function(log_appointment, cancel_on_interruption=False)

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
        # Kick off the conversation:
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
