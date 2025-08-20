import os
import warnings
from typing import List, Optional, Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import after setting up logging
import logging
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import numpy as np
from config import settings

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.QDRANT_COLLECTION
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize Qdrant collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text": models.VectorParams(
                        size=1536,  # OpenAI embedding size
                        distance=models.Distance.COSINE
                    )
                },
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        if not documents:
            return

        # Generate unique IDs for each document
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Get the text content for each document
        texts = [doc.page_content for doc in documents]
        
        # Get embeddings for the documents
        embeddings = self.embeddings.embed_documents(texts)
        
        # Prepare points for batch upload
        points = []
        for idx, (text, doc) in enumerate(zip(texts, documents)):
            point = models.PointStruct(
                id=ids[idx],
                vector={"text": embeddings[idx]},
                payload={
                    "text": text,
                    **doc.metadata,
                    "source": doc.metadata.get("source", "unknown")
                }
            )
            points.append(point)
        
        # Upload to Qdrant in batches
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=("text", query_embedding),
            limit=k,
            with_vectors=False,
            with_payload=True,
        )
        
        # Convert to LangChain documents
        documents = []
        for result in search_result:
            payload = result.payload
            if payload and isinstance(payload, dict):
                doc = Document(
                    page_content=payload.get("text", ""),
                    metadata={
                        k: v for k, v in payload.items() 
                        if k != "text"
                    }
                )
                documents.append(doc)
        
        return documents

    def as_retriever(self):
        """Return a retriever for similarity search"""
        # Create a custom retriever that uses our similarity search
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.pydantic_v1 import Field
        from typing import List, Any
        
        class CustomRetriever(BaseRetriever):
            """Custom retriever that uses our vector store for similarity search."""
            vector_store: Any = Field(..., description="The vector store to use for retrieval")
            
            class Config:
                arbitrary_types_allowed = True
            
            def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
                return self.vector_store.similarity_search(query, k=4)
        
        return CustomRetriever(vector_store=self)

    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Collection '{self.collection_name}' deleted.")
