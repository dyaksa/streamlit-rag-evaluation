from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks
