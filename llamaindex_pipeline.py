"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
#from schemas import OpenAIChatMessage
#from llama_index.embeddings.ollama import OllamaEmbedding
#from llama_index.llms.ollama import Ollama
# from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import Settings
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.llms.openai_like import OpenAILike



class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        import os

        # Set the OpenAI API key
        #os.environ["OPENAI_API_KEY"] = "your-api-key-here"


        print("******* INSIDE PIPELINE LLAMA_INDEX *********")
        

        # Settings.embed_model = OllamaEmbedding(
        #     model_name="avr/sfr-embedding-mistral:f16",
        #     base_url="http://localhost:11434",
        # )
        # # Settings.llm = OpenAILike(model="meta-llama/Meta-Llama-3-70B-Instruct", api_base="http://llama3-70b.e4-analytics.lan:8000/v1", api_key="fake",
        # #                           system_prompt="""You are an agent designed to answer queries over a single given paper. Do not rely on prior knowledge.
        # #                                             Please be as much detailed as possible."""
        # #                         )
        # Settings.llm = Ollama(model="mixtral:8x22b-instruct-v0.1-fp16")

        # splitter = SentenceSplitter(chunk_size=1024)

        # self.documents = SimpleDirectoryReader("./pipelines/data").load_data()
        # nodes = splitter.get_nodes_from_documents(self.documents)
        # self.index = VectorStoreIndex(nodes)

        #self.index = VectorStoreIndex.from_documents(self.documents)
        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.
        print("******INSIDE PIPELINE******")
        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
