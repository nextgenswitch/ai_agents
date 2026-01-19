import argparse
import sys
from contextlib import asynccontextmanager

import os
from typing import Any, Dict
import uvicorn
import json
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import run_bot

# Load environment variables
load_dotenv(override=True)

app = FastAPI()

# Initialize the SmallWebRTC request handler
small_webrtc_handler: SmallWebRTCRequestHandler = SmallWebRTCRequestHandler()


async def _initialize_websocket(websocket: WebSocket) -> Dict[str, Any]:
    await websocket.accept()
    start_iterator = websocket.iter_text()

    try:
        start_message = await start_iterator.__anext__()
        logger.debug("Received websocket start frame: {}", start_message)
        raw_payload = await start_iterator.__anext__()
    except StopAsyncIteration as exc:
        logger.warning("Websocket disconnected before sending call data")
        raise WebSocketDisconnect(code=1002) from exc

    try:
        call_data = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON payload from websocket: {}", raw_payload)
        await websocket.close(code=1003)
        raise WebSocketDisconnect(code=1003) from exc

    return call_data

@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Handle WebRTC offer requests via SmallWebRTCRequestHandler."""

    # Prepare runner arguments with the callback to run your bot
    async def webrtc_connection_callback(connection):
        background_tasks.add_task(run_bot, webrtc_connection=connection)

    # Delegate handling to SmallWebRTCRequestHandler
    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    logger.debug(f"Received patch request: {request}")
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def _websocket(websocket: WebSocket) -> None:
    try:
        call_data = await _initialize_websocket(websocket)
    except WebSocketDisconnect:
        logger.warning("Gemini websocket initialization failed")
        raise

    stream_id = call_data.get("streamId") or call_data.get("streamSid") or os.getenv("DEFAULT_STREAM_ID", "pipecat-test-stream")
    call_sid = call_data.get("call_id") or call_data.get("callSid") or os.getenv("DEFAULT_CALL_SID", "pipecat-test-call")

    bot_params = call_data.get("params") or {}
    if not isinstance(bot_params, dict):
        logger.warning("Gemini params payload must be a dict. Received {}", bot_params)
        bot_params = {}

    logger.info(
        "Gemini websocket connected stream={} call={}  params_keys={}",
        stream_id,
        call_sid,
        list(bot_params.keys()),
    )

    await run_bot(
        websocket=websocket,
        stream_id=stream_id,
        call_sid=call_sid,
        bot_params=bot_params,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    await small_webrtc_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Receptionist HTTP server.")
    parser.add_argument(
        "--host", default=os.getenv("HOST", "0.0.0.0"), help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=os.getenv("PORT", 7860), help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)
