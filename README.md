# AI Agents

A collection of voice-first agents built with NextGenSwitch + Pipecat and deployed on Cerebrium.
Each project has its own code, environment and `cerebrium.toml`.

## Projects

### CareDesk

- Location: `CareDesk/`
- Summary: Medical clinic appointment receptionist that books, reschedules, and cancels visits, and logs to an Excel.
- Stack: OpenAI + Deepgram STT + Cartesia TTS + NextGenSwitch + Pipecat.
- [Demo live link](https://api.aws.us-east-1.cerebrium.ai/v4/p-9d2ce1e5/caredesk/)
- use the websocket endpoint `wss://api.aws.us-east-1.cerebrium.ai/v4/p-9d2ce1e5/caredesk/ws` to connect with NextGenSwitch

### Receptionist Agent

- Location: `Receptionist/`
- Summary: Voice receptionist for call intake and transfer workflows.
- Stack: Google Gemini + NextGenSwitch + Pipecat.
- [Demo Live Link](https://api.aws.us-east-1.cerebrium.ai/v4/p-9d2ce1e5/receptionist-agent/)
- use the websocket endpoint `wss://api.aws.us-east-1.cerebrium.ai/v4/p-9d2ce1e5/receptionist-agent/ws` to connect with NextGenSwitch
