from curses import meta
import chromadb
from chromadb.errors import NotFoundError
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
import Stemmer


def create_vector_store_index(
    collection_name, documents, embedding
) -> VectorStoreIndex:
    Settings.embed_model = embedding
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(name=collection_name)
    except NotFoundError:
        pass
    except Exception as e:
        print(f"Error occurred while creating vector store index: {e}")

    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(
        chroma_collection=collection,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = [Document(text=doc) for doc in documents]
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )

    return index


def create_bm25_retriever(documents, top_k: int = 3) -> BM25Retriever:
    documents = [Document(text=doc) for doc in documents]
    nodes = SentenceSplitter().get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    return bm25_retriever


def create_query_fusion_retriever(
    documents, embedding, llm, top_k: int = 3
) -> QueryFusionRetriever:
    Settings.embed_model = embedding
    Settings.llm = llm
    documents = [Document(text=doc) for doc in documents]
    nodes = SentenceSplitter().get_nodes_from_documents(documents)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=top_k)

    rrf_query_fusion = QueryFusionRetriever(
        retrievers=[bm25_retriever, retriever],
        similarity_top_k=3,
        num_queries=3,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
        verbose=True,
    )

    return rrf_query_fusion


def retrieve(
    index: VectorStoreIndex | BM25Retriever | QueryFusionRetriever,
    query: str,
    top_k: int = 3,
):
    if isinstance(index, VectorStoreIndex):
        query_engine = index.as_retriever(similarity_top_k=top_k)
        response = query_engine.retrieve(query)
        return response

    if isinstance(index, BM25Retriever):
        response = index.retrieve(query)
        return response

    if isinstance(index, QueryFusionRetriever):
        response = index.retrieve(query)
        return response

    raise ValueError("Index must be either VectorStoreIndex or BM25Retriever.")
