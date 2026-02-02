import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


def _optional(value):
    if value is None or value == "":
        return None
    return value


def _add_optional(kwargs, key, value):
    value = _optional(value)
    if value is not None:
        kwargs[key] = value


def get_llm_service(bot_params: dict):
    provider = (
        bot_params.get("llm_provider")
        or os.getenv("LLM_PROVIDER", "openai")
    )
    provider = provider.lower()

    if provider in {"openai", "open_ai"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing OpenAI API key")

        model = (
            bot_params.get("model")
            or os.getenv("OPENAI_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
        )
        params = bot_params.get("params")

        from pipecat.services.openai.llm import OpenAILLMService

        logger.info("Using OpenAI LLM Service")
        kwargs = {"api_key": api_key}
        _add_optional(kwargs, "model", model)
        _add_optional(kwargs, "base_url", base_url)
        if params is not None:
            kwargs["params"] = params
        return OpenAILLMService(**kwargs)

    elif provider in {"anthropic", "claude"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Anthropic API key")

        model = (
            bot_params.get("model")
            or os.getenv("ANTHROPIC_MODEL")
        )
        params = bot_params.get("params")

        from pipecat.services.anthropic.llm import AnthropicLLMService

        logger.info("Using Anthropic LLM Service")
        kwargs = {"api_key": api_key}
        _add_optional(kwargs, "model", model)
        if params is not None:
            kwargs["params"] = params
        return AnthropicLLMService(**kwargs)

    elif provider in {"azure", "azure_openai", "azure_chatgpt"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("AZURE_CHATGPT_API_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Azure OpenAI API key")

        endpoint = (
            bot_params.get("endpoint")
            or os.getenv("AZURE_CHATGPT_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        if not endpoint:
            raise ValueError("Missing Azure OpenAI endpoint")

        model = (
            bot_params.get("model")
            or os.getenv("AZURE_CHATGPT_MODEL")
            or os.getenv("AZURE_OPENAI_MODEL")
        )
        if not model:
            raise ValueError("Missing Azure OpenAI model")

        api_version = (
            bot_params.get("api_version")
            or os.getenv("AZURE_OPENAI_API_VERSION")
        )
        params = bot_params.get("params")

        from pipecat.services.azure.llm import AzureLLMService

        logger.info("Using Azure OpenAI LLM Service")
        kwargs = {
            "api_key": api_key,
            "endpoint": endpoint,
            "model": model,
        }
        _add_optional(kwargs, "api_version", api_version)
        if params is not None:
            kwargs["params"] = params
        return AzureLLMService(**kwargs)

    elif provider in {"deepseek", "deep_seek"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("DEEPSEEK_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing DeepSeek API key")

        model = (
            bot_params.get("model")
            or os.getenv("DEEPSEEK_MODEL")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("DEEPSEEK_BASE_URL")
        )
        params = bot_params.get("params")

        from pipecat.services.deepseek.llm import DeepSeekLLMService

        logger.info("Using DeepSeek LLM Service")
        kwargs = {"api_key": api_key}
        _add_optional(kwargs, "model", model)
        _add_optional(kwargs, "base_url", base_url)
        if params is not None:
            kwargs["params"] = params
        return DeepSeekLLMService(**kwargs)

    elif provider in {"gemini", "google", "google_gemini"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Google API key")

        model = (
            bot_params.get("model")
            or os.getenv("GEMINI_MODEL")
            or os.getenv("GOOGLE_MODEL")
        )
        params = bot_params.get("params")
        system_instruction = bot_params.get("system_instruction")
        tools = bot_params.get("tools")
        tool_config = bot_params.get("tool_config")
        http_options = bot_params.get("http_options")

        from pipecat.services.google.llm import GoogleLLMService

        logger.info("Using Google Gemini LLM Service")
        kwargs = {"api_key": api_key}
        _add_optional(kwargs, "model", model)
        if params is not None:
            kwargs["params"] = params
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        if tools is not None:
            kwargs["tools"] = tools
        if tool_config is not None:
            kwargs["tool_config"] = tool_config
        if http_options is not None:
            kwargs["http_options"] = http_options
        return GoogleLLMService(**kwargs)

    elif provider in {"grok", "xai"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("XAI_API_KEY")
            or os.getenv("GROK_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Grok (xAI) API key")

        model = (
            bot_params.get("model")
            or os.getenv("XAI_MODEL")
            or os.getenv("GROK_MODEL")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("XAI_BASE_URL")
            or os.getenv("GROK_BASE_URL")
        )
        params = bot_params.get("params")

        from pipecat.services.grok.llm import GrokLLMService

        logger.info("Using Grok LLM Service")
        kwargs = {"api_key": api_key}
        _add_optional(kwargs, "model", model)
        _add_optional(kwargs, "base_url", base_url)
        if params is not None:
            kwargs["params"] = params
        return GrokLLMService(**kwargs)

    elif provider == "groq":
        api_key = (
            bot_params.get("api_key")
            or os.getenv("GROQ_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Groq API key")

        model = (
            bot_params.get("model")
            or os.getenv("GROQ_MODEL")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("GROQ_BASE_URL")
        )
        params = bot_params.get("params")

        from pipecat.services.groq.llm import GroqLLMService

        logger.info("Using Groq LLM Service")
        kwargs = {"api_key": api_key}
        _add_optional(kwargs, "model", model)
        _add_optional(kwargs, "base_url", base_url)
        if params is not None:
            kwargs["params"] = params
        return GroqLLMService(**kwargs)

    elif provider == "ollama":
        model = (
            bot_params.get("model")
            or os.getenv("OLLAMA_MODEL")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("OLLAMA_BASE_URL")
        )
        params = bot_params.get("params")

        from pipecat.services.ollama.llm import OLLamaLLMService

        logger.info("Using Ollama LLM Service")
        kwargs = {}
        _add_optional(kwargs, "model", model)
        _add_optional(kwargs, "base_url", base_url)
        if params is not None:
            kwargs["params"] = params
        return OLLamaLLMService(**kwargs)

    elif provider in {"openrouter", "open_router"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("OPENROUTER_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing OpenRouter API key")

        model = (
            bot_params.get("model")
            or os.getenv("OPENROUTER_MODEL")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("OPENROUTER_BASE_URL")
        )
        params = bot_params.get("params")

        from pipecat.services.openrouter.llm import OpenRouterLLMService

        logger.info("Using OpenRouter LLM Service")
        kwargs = {"api_key": api_key}
        _add_optional(kwargs, "model", model)
        _add_optional(kwargs, "base_url", base_url)
        if params is not None:
            kwargs["params"] = params
        return OpenRouterLLMService(**kwargs)

    elif provider in {"perplexity", "pplx"}:
        api_key = (
            bot_params.get("api_key")
            or os.getenv("PPLX_API_KEY")
            or os.getenv("PERPLEXITY_API_KEY")
        )
        if not api_key:
            raise ValueError("Missing Perplexity API key")

        model = (
            bot_params.get("model")
            or os.getenv("PERPLEXITY_MODEL")
        )
        base_url = (
            bot_params.get("base_url")
            or os.getenv("PERPLEXITY_BASE_URL")
        )
        params = bot_params.get("params")

        from pipecat.services.perplexity.llm import PerplexityLLMService

        logger.info("Using Perplexity LLM Service")
        kwargs = {"api_key": api_key}
        _add_optional(kwargs, "model", model)
        _add_optional(kwargs, "base_url", base_url)
        if params is not None:
            kwargs["params"] = params
        return PerplexityLLMService(**kwargs)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
