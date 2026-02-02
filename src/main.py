import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import os
from typing import Any, Dict, List
import uvicorn
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel
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

# Agents directory
AGENTS_DIR = Path(__file__).parent / "agents"


class AgentConfig(BaseModel):
    id: str = None
    name: str
    description: str = ""
    prompt: str
    greeting_message: str
    stt_provider: str = "deepgram"
    tts_provider: str = "deepgram"
    llm_provider: str = "openai"
    icon: str = "microphone"
    color: str = "purple"


def get_agent_config(agent: str) -> Dict[str, Any]:
    """Load agent configuration from agents directory or return default config."""
    agent_config = None
    agent_file = AGENTS_DIR / f"{agent}.json"
    
    # Try exact match first, then search all files
    if agent_file.exists():
        with open(agent_file, "r") as f:
            agent_config = json.load(f)
    else:
        # Search for agent by ID in all JSON files
        for file in AGENTS_DIR.glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)
                if data.get("id") == agent:
                    agent_config = data
                    break
    
    if not agent_config:
        logger.warning(f"Agent config not found for: {agent}, using defaults")
        agent_config = {
            "id": agent,
            "name": agent.title(),
            "prompt": "You are a helpful AI voice assistant.",
            "greeting_message": "Hello! How can I help you today?",
            "llm_provider": "openai",
            "stt_provider": "deepgram",
            "tts_provider": "deepgram"
        }
    
    logger.info(f"Loaded agent config: {agent_config.get('name', agent)}")
    return agent_config

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
async def offer(
    request: SmallWebRTCRequest,
    background_tasks: BackgroundTasks,
    agent: str = "receptionist",
):
    """Handle WebRTC offer requests via SmallWebRTCRequestHandler."""
    
    agent_config = get_agent_config(agent)

    # Prepare runner arguments with the callback to run your bot
    async def webrtc_connection_callback(connection):
        bot_params = {
            "agent": agent,
            "prompt": agent_config.get("prompt"),
            "greetings": agent_config.get("greeting_message"),
            "llm_provider": agent_config.get("llm_provider", "openai"),
            "stt_provider": agent_config.get("stt_provider", "deepgram"),
            "tts_provider": agent_config.get("tts_provider", "deepgram"),
        }
        background_tasks.add_task(run_bot, webrtc_connection=connection, bot_params=bot_params)

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


# Agent management API endpoints
@app.get("/api/agents")
async def list_agents() -> List[dict]:
    """List all agents from the agents directory, sorted by creation date (latest first)."""
    agents = []
    if AGENTS_DIR.exists():
        for file_path in AGENTS_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    agent_data = json.load(f)
                    agent_data["filename"] = file_path.name
                    # Use file modification time as fallback if created_at is not present
                    if "created_at" not in agent_data:
                        agent_data["created_at"] = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    agents.append(agent_data)
            except Exception as e:
                logger.error(f"Failed to load agent {file_path}: {e}")
    # Sort by created_at in descending order (latest first)
    agents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return agents


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str) -> dict:
    """Get a specific agent by ID."""
    if AGENTS_DIR.exists():
        for file_path in AGENTS_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    agent_data = json.load(f)
                    if agent_data.get("id") == agent_id:
                        agent_data["filename"] = file_path.name
                        return agent_data
            except Exception as e:
                logger.error(f"Failed to load agent {file_path}: {e}")
    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/api/agents")
async def create_agent(agent: AgentConfig) -> dict:
    """Create a new agent and save to the agents directory."""
    # Create agents directory if it doesn't exist
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate ID and filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_id = f"{agent.name.lower().replace(' ', '_')}_{timestamp}"
    filename = f"{agent_id}.json"
    file_path = AGENTS_DIR / filename
    
    # Prepare agent data
    agent_data = agent.model_dump()
    agent_data["id"] = agent_id
    agent_data["created_at"] = datetime.now().isoformat()
    
    # Save to file
    try:
        with open(file_path, "w") as f:
            json.dump(agent_data, f, indent=2)
        agent_data["filename"] = filename
        logger.info(f"Created agent: {filename}")
        return agent_data
    except Exception as e:
        logger.error(f"Failed to save agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to save agent")


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str) -> dict:
    """Delete an agent by ID."""
    if AGENTS_DIR.exists():
        for file_path in AGENTS_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    agent_data = json.load(f)
                    if agent_data.get("id") == agent_id:
                        file_path.unlink()
                        logger.info(f"Deleted agent: {file_path.name}")
                        return {"status": "deleted", "id": agent_id}
            except Exception as e:
                logger.error(f"Failed to process agent {file_path}: {e}")
    raise HTTPException(status_code=404, detail="Agent not found")


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
    agent = bot_params.get("agent") if isinstance(bot_params, dict) else None
    agent_config = get_agent_config(agent) if agent else None
    if agent_config:
        bot_params.setdefault("prompt", agent_config.get("prompt"))
        bot_params.setdefault("greetings", agent_config.get("greeting_message"))
        bot_params.setdefault("llm_provider", agent_config.get("llm_provider", "openai"))
        bot_params.setdefault("stt_provider", agent_config.get("stt_provider", "deepgram"))
        bot_params.setdefault("tts_provider", agent_config.get("tts_provider", "deepgram"))
        
    if not isinstance(bot_params, dict):
        logger.warning("Gemini params payload must be a dict. Received {}", bot_params)
        bot_params = {}

    logger.info(
        "websocket connected stream={} call={}  params_keys={}",
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
    parser = argparse.ArgumentParser(description="Run the CareDesk FastAPI server.")
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
