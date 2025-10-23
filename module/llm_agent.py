from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.gemini import Gemini
from module.config import settings


def gemini_llm(
    temperature: float = 0.1,
    max_retries: int = 3,
    max_tokens: int = 500,
) -> Gemini:
    llm = Gemini(
        model=settings.GEMINI_MODEL_LLM,
        api_key=settings.GEMINI_API_KEY,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=temperature,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )

    return llm


def huggingface_llm(
    temperature: float = 0.1,
    max_retries: int = 3,
    max_tokens: int = 500,
) -> OpenAILike:
    llm = OpenAILike(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key=settings.HUGGINGFACE_API_KEY,
        is_chat_model=True,
        is_function_calling_model=False,
        api_base="https://router.huggingface.co/v1",
        temperature=temperature,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )

    return llm


def get_llm(
    temperature: float = 0.1, max_retries: int = 3, max_tokens: int = 500
) -> OpenRouter:
    llm = OpenRouter(
        model=settings.OPEN_ROUTER_MODEL,
        api_key=settings.OPEN_ROUTER_API_KEY,
        base_url=settings.OPEN_ROUTER_BASE_URL,
        temperature=temperature,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )

    return llm
