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


def get_stt_service(bot_params: dict):
    provider = (
        bot_params.get("stt_provider")
        or os.getenv("STT_PROVIDER", "deepgram")
    )
    provider = provider.lower()

    if provider in {"amazon_transcribe", "aws_transcribe", "aws"}:
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

        language = bot_params.get("language")
        if language is None:
            from pipecat.transcriptions.language import Language
            language = Language.EN
        else:
            from pipecat.transcriptions.language import Language
            if isinstance(language, str):
                try:
                    language = Language(language)
                except ValueError:
                    language = Language.EN

        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("AWS_SAMPLE_RATE")
        )

        from pipecat.services.aws.stt import AWSTranscribeSTTService

        logger.info("Using AWS Transcribe STT Service")
        return AWSTranscribeSTTService(
            api_key=secret_access_key,
            aws_access_key_id=access_key_id,
            aws_session_token=session_token,
            region=region,
            language=language,
            sample_rate=sample_rate or 16000,
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

        endpoint_id = (
            bot_params.get("endpoint_id")
            or os.getenv("AZURE_SPEECH_ENDPOINT_ID")
        )

        language = (
            bot_params.get("language")
            or os.getenv("AZURE_SPEECH_LANGUAGE")
            or os.getenv("AZURE_LANGUAGE")
        )
        from pipecat.transcriptions.language import Language
        if language is None:
            language = Language.EN_US
        elif isinstance(language, str):
            try:
                language = Language(language)
            except ValueError:
                language = Language.EN_US
        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("AZURE_SAMPLE_RATE")
        )

        from pipecat.services.azure.stt import AzureSTTService

        logger.info("Using Azure STT Service")
        kwargs = {
            "api_key": api_key,
            "region": region,
            "language": language,
            "endpoint_id": endpoint_id,
        }
        if sample_rate is not None:
            kwargs["sample_rate"] = sample_rate
        return AzureSTTService(
            **kwargs,
        )

    elif provider == "cartesia":
        api_key = (
            bot_params.get("api_key")
            or os.getenv("CARTESIA_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Cartesia API key")

        base_url = (
            bot_params.get("base_url")
            or os.getenv("CARTESIA_BASE_URL")
        )
        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("CARTESIA_SAMPLE_RATE")
        )
        live_options = bot_params.get("live_options")

        from pipecat.services.cartesia.stt import (
            CartesiaLiveOptions,
            CartesiaSTTService,
        )

        if isinstance(live_options, dict):
            live_options = CartesiaLiveOptions(**live_options)

        logger.info("Using Cartesia STT Service")
        kwargs = {
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if live_options is not None:
            kwargs["live_options"] = live_options
        if sample_rate:
            kwargs["sample_rate"] = sample_rate
        return CartesiaSTTService(**kwargs)

    elif provider == "deepgram":
        api_key = (
            bot_params.get("api_key")
            or os.getenv("DEEPGRAM_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Deepgram API key")

        base_url = (
            bot_params.get("base_url")
            or os.getenv("DEEPGRAM_BASE_URL")
        )
        url = (
            bot_params.get("url")
            or os.getenv("DEEPGRAM_URL")
        )
        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("DEEPGRAM_SAMPLE_RATE")
        )
        live_options = bot_params.get("live_options")
        addons = bot_params.get("addons")
        should_interrupt = bot_params.get("should_interrupt")

        from pipecat.services.deepgram.stt import DeepgramSTTService

        logger.info("Using Deepgram STT Service")
        kwargs = {
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if url:
            kwargs["url"] = url
        if live_options is not None:
            kwargs["live_options"] = live_options
        if addons is not None:
            kwargs["addons"] = addons
        if sample_rate:
            kwargs["sample_rate"] = sample_rate
        if should_interrupt is not None:
            kwargs["should_interrupt"] = should_interrupt
        return DeepgramSTTService(**kwargs)

    elif provider == "elevenlabs":
        api_key = (
            bot_params.get("api_key")
            or os.getenv("ELEVENLABS_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing ElevenLabs API key")

        aiohttp_session = bot_params.get("aiohttp_session")
        if not aiohttp_session:
            raise ValueError("Missing aiohttp_session for ElevenLabs STT")

        model = (
            bot_params.get("model")
            or os.getenv("ELEVENLABS_STT_MODEL")
            or "scribe_v1"
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("ELEVENLABS_BASE_URL")
            or "https://api.elevenlabs.io"
        )
        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("ELEVENLABS_SAMPLE_RATE")
        )
        params = bot_params.get("params")

        from pipecat.services.elevenlabs.stt import ElevenLabsSTTService

        logger.info("Using ElevenLabs STT Service")
        kwargs = {
            "api_key": api_key,
            "aiohttp_session": aiohttp_session,
            "model": model,
            "base_url": base_url,
            "params": params,
        }
        if sample_rate:
            kwargs["sample_rate"] = sample_rate
        return ElevenLabsSTTService(**kwargs)

    elif provider in {"google", "google_stt", "google_cloud"}:
        credentials = (
            bot_params.get("credentials")
            or os.getenv("GOOGLE_CREDENTIALS_JSON")
        )
        credentials_path = (
            bot_params.get("credentials_path")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        location = (
            bot_params.get("location")
            or os.getenv("GOOGLE_STT_LOCATION")
            or "global"
        )
        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("GOOGLE_SAMPLE_RATE")
        )
        params = bot_params.get("params")

        from pipecat.services.google.stt import GoogleSTTService

        logger.info("Using Google STT Service")
        kwargs = {
            "credentials": credentials,
            "credentials_path": credentials_path,
            "location": location,
            "params": params,
        }
        if sample_rate:
            kwargs["sample_rate"] = sample_rate
        return GoogleSTTService(**kwargs)

    elif provider in {"openai", "openai_whisper", "whisper"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing OpenAI API key")

        model = (
            bot_params.get("model")
            or os.getenv("OPENAI_STT_MODEL")
            or "gpt-4o-transcribe"
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
        )
        language = bot_params.get("language")
        prompt = bot_params.get("prompt")
        temperature = bot_params.get("temperature")

        from pipecat.services.openai.stt import OpenAISTTService

        logger.info("Using OpenAI STT Service")
        kwargs = {
            "api_key": api_key,
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if language is not None:
            kwargs["language"] = language
        return OpenAISTTService(**kwargs)

    elif provider in {"fal", "fal_wizper", "wizper"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("FAL_KEY")
        )
        if not api_key:
            raise ValueError("Missing FAL API key")

        sample_rate = _coerce_int(
            bot_params.get("sample_rate")
            or os.getenv("FAL_SAMPLE_RATE")
        )
        params = bot_params.get("params")

        from pipecat.services.fal.stt import FalSTTService

        logger.info("Using FAL STT Service")
        kwargs = {
            "api_key": api_key,
            "params": params,
        }
        if sample_rate:
            kwargs["sample_rate"] = sample_rate
        return FalSTTService(**kwargs)

    else:
        raise ValueError(f"Unsupported STT provider: {provider}")
