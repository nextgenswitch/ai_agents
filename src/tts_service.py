import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


def _coerce_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_tts_service(bot_params: dict):
    provider = (
        bot_params.get("tts_provider")
        or os.getenv("TTS_PROVIDER", "deepgram")
    )
    provider = provider.lower()

    if provider in {"amazon_polly", "aws_polly", "aws"}:
        secret_access_key = (
            bot_params.get("secret_access_key")
            or bot_params.get("api_key")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        if not secret_access_key:
            raise ValueError("Missing AWS secret access key")

        access_key_id = (
            bot_params.get("access_key_id")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        if not access_key_id:
            raise ValueError("Missing AWS access key ID")

        session_token = (
            bot_params.get("session_token")
            or os.getenv("AWS_SESSION_TOKEN")
        )
        region = (
            bot_params.get("region")
            or os.getenv("AWS_REGION")
            or "us-east-1"
        )
        voice_id = (
            bot_params.get("voice_id")
            or os.getenv("AWS_VOICE_ID")
            or "Joanna"
        )
        engine = (
            bot_params.get("engine")
            or os.getenv("AWS_TTS_ENGINE")
            or "generative"
        )
        rate = (
            bot_params.get("rate")
            or os.getenv("AWS_TTS_RATE")
            or "1.1"
        )
        params = bot_params.get("params")

        from pipecat.services.aws.tts import AWSPollyTTSService

        if params is None:
            params = AWSPollyTTSService.InputParams(
                engine=engine,
                rate=rate,
            )

        logger.info("Using Amazon Polly TTS Service")
        return AWSPollyTTSService(
            api_key=secret_access_key,
            aws_access_key_id=access_key_id,
            aws_session_token=session_token,
            region=region,
            voice_id=voice_id,
            params=params,
        )

    elif provider == "azure":
        api_key = (
            bot_params.get("api_key")
            or os.getenv("AZURE_SPEECH_API_KEY")
            or os.getenv("AZURE_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Azure API key")

        region = (
            bot_params.get("region")
            or os.getenv("AZURE_SPEECH_REGION")
            or os.getenv("AZURE_REGION")
        )
        if not region:
            raise ValueError("Missing Azure region")

        voice_id = (
            bot_params.get("voice_id")
            or os.getenv("AZURE_SPEECH_VOICE_ID")
            or os.getenv("AZURE_VOICE_ID")
            or "en-US-JennyNeural"
        )
        language = (
            bot_params.get("language")
            or os.getenv("AZURE_SPEECH_LANGUAGE")
            or os.getenv("AZURE_LANGUAGE")
        )
        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("AZURE_SAMPLE_RATE")
        )
        params = bot_params.get("params")

        from pipecat.services.azure.tts import AzureTTSService

        logger.info("Using Azure TTS Service")
        kwargs = {
            "api_key": api_key,
            "region": region,
            "voice_id": voice_id,
        }
        if language:
            kwargs["language"] = language
        if sample_rate is not None:
            kwargs["sample_rate"] = sample_rate
        if params is not None:
            kwargs["params"] = params
        return AzureTTSService(**kwargs)

    elif provider == "cartesia":
        api_key = (
            bot_params.get("api_key")
            or os.getenv("CARTESIA_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Cartesia API key")

        voice_id = (
            bot_params.get("voice_id")
            or os.getenv("CARTESIA_VOICE_ID")
            or "71a7ad14-091c-4e8e-a314-022ece01c121"
        )
        model_id = (
            bot_params.get("model_id")
            or os.getenv("CARTESIA_MODEL_ID")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("CARTESIA_BASE_URL")
        )
        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("CARTESIA_SAMPLE_RATE")
        )
        params = bot_params.get("params")

        from pipecat.services.cartesia.tts import CartesiaTTSService

        logger.info("Using Cartesia TTS Service")
        kwargs = {
            "api_key": api_key,
            "voice_id": voice_id,
        }
        if model_id:
            kwargs["model_id"] = model_id
        if base_url:
            kwargs["base_url"] = base_url
        if sample_rate:
            kwargs["sample_rate"] = sample_rate
        if params is not None:
            kwargs["params"] = params
        return CartesiaTTSService(**kwargs)

    elif provider == "deepgram":
        api_key = (
            bot_params.get("api_key")
            or os.getenv("DEEPGRAM_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Deepgram API key")

        voice_id = (
            bot_params.get("voice_id")
            or os.getenv("DEEPGRAM_VOICE_ID")
            or "aura-2-athena-en"
        )
        model = (
            bot_params.get("model")
            or os.getenv("DEEPGRAM_TTS_MODEL")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("DEEPGRAM_BASE_URL")
        )
        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("DEEPGRAM_SAMPLE_RATE")
        )
        params = bot_params.get("params")

        from pipecat.services.deepgram.tts import DeepgramTTSService

        logger.info("Using Deepgram TTS Service")
        kwargs = {
            "api_key": api_key,
            "voice_id": voice_id,
        }
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        if sample_rate:
            kwargs["sample_rate"] = sample_rate
        if params is not None:
            kwargs["params"] = params
        return DeepgramTTSService(**kwargs)

    elif provider == "elevenlabs":
        api_key = (
            bot_params.get("api_key")
            or os.getenv("ELEVENLABS_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing ElevenLabs API key")

        voice_id = (
            bot_params.get("voice_id")
            or os.getenv("ELEVENLABS_VOICE_ID")
            or "21m00Tcm4TlvDq8ikWAM"
        )
        model_id = (
            bot_params.get("model_id")
            or os.getenv("ELEVENLABS_MODEL_ID")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("ELEVENLABS_BASE_URL")
        )
        output_format = (
            bot_params.get("output_format")
            or os.getenv("ELEVENLABS_OUTPUT_FORMAT")
        )
        params = bot_params.get("params")

        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        logger.info("Using ElevenLabs TTS Service")
        kwargs = {
            "api_key": api_key,
            "voice_id": voice_id,
        }
        if model_id:
            kwargs["model_id"] = model_id
        if base_url:
            kwargs["base_url"] = base_url
        if output_format:
            kwargs["output_format"] = output_format
        if params is not None:
            kwargs["params"] = params
        return ElevenLabsTTSService(**kwargs)

    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")
