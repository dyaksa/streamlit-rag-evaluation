from module.config import settings
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding


def get_embedding_huggingface() -> HuggingFaceInferenceAPIEmbedding:
    embedding = HuggingFaceInferenceAPIEmbedding(
        model_name=settings.HUGGINGFACE_EMBEDDING_MODEL,
        token=settings.HUGGINGFACE_API_KEY,
    )

    return embedding


def get_embedding_gemini() -> GeminiEmbedding:
    embedding = GeminiEmbedding(
        api_key=settings.GEMINI_API_KEY,
    )

    return embedding
