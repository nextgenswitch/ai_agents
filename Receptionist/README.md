# Receptionist Agent

A voice-first receptionist agent built with NextGenSwitch and Pipecat.

## Local run

1. Install dependencies

```bash
cd Receptionist
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment

```bash
cp env.example .env
# edit .env with necessary values
```

3. Start the server

```bash
python main.py --host 0.0.0.0 --port 7860
```

4. Open the WebRTC UI

- Visit `http://localhost:7860/` and click Connect.

## Cerebrium deployment

This repo includes `cerebrium.toml` with the deployment configuration.

1. Set the required environment variables in Cerebrium:

- GOOGLE_API_KEY=
- FORWARDING_NUMBER=
- NEXTGENSWITCH_URL=
- NEXTGENSWITCH_API_KEY=
- NEXTGENSWITCH_API_SECRET=

2. deploy using the Cerebrium CLI (example):

```bash
cerebrium login
cerebrium deploy
```

## Environment variables

Required:

- `GOOGLE_API_KEY`: Google Gemini API key.

Optional server settings:

- `HOST`: server host (default `localhost` when running `main.py`).
- `PORT`: server port (default `7860`).

NextGenSwitch defaults and call transfer:

- `NEXTGENSWITCH_URL`: base API URL for transfer (e.g. `https://sg.nextgenswitch.com/api/v1`).
- `NEXTGENSWITCH_API_KEY`: API key for transfer.
- `NEXTGENSWITCH_API_SECRET`: API secret for transfer.
- `FORWARDING_NUMBER`: optional default number for transfer workflows.
