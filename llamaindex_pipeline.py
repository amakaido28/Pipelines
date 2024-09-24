"""
title: Llama Index Pipeline Test
author: E4 Analytics
date: 23-09-2024
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index, chromadb, requests, llama-index-vector-stores-chroma, llama-index-embeddings-text-embeddings-inference, llama-index-llms-openai-like
"""

from typing import List, Union, Generator, Iterator
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
from llama_index.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import RetrieverQueryEngine
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from llama_index.core.response_synthesizers.type import ResponseMode
from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        VLLM_BASE_URL: str
        VLLM_API_KEY: str
        TEI_BASE_URL: str
        TEI_API_KEY: str
        CHROMA_BASE_URL: str
        CHROMA_PORT: str

    def __init__(self):
        # Optionally, you can set the name of the manifold pipeline.
        self.name = "pipeline_chroma_llamaindex"
        self.collection_name = ""

        # Initialize rate limits
        self.valves = self.Valves(
            **{
                "VLLM_BASE_URL": os.getenv(
                    "VLLM_BASE_URL", "http://172.18.21.141:8000/v1"
                ),
                "VLLM_API_KEY": os.getenv("VLLM_API_KEY", "empty"),
                "TEI_BASE_URL": os.getenv(
                    "TEI_BASE_URL", "http://172.18.21.144:80"
                ),
                "TEI_API_KEY": os.getenv("TEI_API_KEY", "empty"),
                "CHROMA_BASE_URL": os.getenv(
                    "CHROMA_BASE_URL", "172.18.21.155"
                ),
                "CHROMA_PORT": os.getenv("CHROMA_PORT", "8000"),
            }
        )
        pass

    async def on_startup(self):
        # This function is called when the server is started

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        print(body)
        print(user)

        collection_names = [file['collection_name'] for file in body['files']]

        self.collection_name = collection_names[0]

        return body

    def pipe(
        self, body:dict, user_message: str, model_id: str, messages: List[dict],
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.

        print(body)

        vector_store = ChromaVectorStore(
            collection_name=self.collection_name,
            host=self.valves.CHROMA_BASE_URL,
            port=self.valves.CHROMA_PORT,
            ssl=False,
            headers={"Authorization": "Bearer fGaO9goSYRbUDPiXSQS5gT3fZverbQqw"},
        )

        embed_model = TextEmbeddingsInference(
            model_name="Salesforce/SFR-Embedding-2_R",
            base_url=self.valves.TEI_BASE_URL,
            timeout=60,
            embed_batch_size=10,
        )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )

        self.retriver = self.index.as_retriever(
            similarity_top_k=10,
            alpha=0.75,
            vector_store_query_mode="hybrid",
            embed_model=embed_model,
        )

        model = {
            "url": self.valves.VLLM_BASE_URL,
            "name": "mistralai/Pixtral-12B-2409",
        }

        temperature = 0
        max_tokens = 1024

        self.llm = OpenAILike(
            model=model["name"],
            api_base=model["url"],
            api_key=self.valves.VLLM_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=600,
        )

        rag_query_engine = RetrieverQueryEngine.from_args(
            llm=self.llm,
            retriever=self.retriver,
            response_mode=ResponseMode.COMPACT,
            streaming=True,
        )
        response = rag_query_engine.query(user_message)

        return response.response_gen
