from pathlib import Path
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from typing import Optional, List, Union

from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

class ConversationManager:
    
    CONFIG_DIR = '.lesa'
    
    def __init__(self, 
                 base_path: str = '.',
                #  output_dir: str = '.', 
                 document_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 generator_model: str = "qwen:4b",
                 generator_url: str = "http://localhost:11434"):
        """
        Initialize the RAG pipeline with configurable parameters.
        
        :param output_dir: Directory containing source documents
        :param embedding_path: Path to store embeddings
        :param document_model: Embedding model for documents
        :param generator_model: LLM model for generating responses
        :param generator_url: URL for the LLM generator
        """
        
        self.base_path = os.path.abspath(base_path)
        self.embedding_path = os.path.join(self.base_path, self.CONFIG_DIR, 'embeddings')
        
        # Initialize document store
        self.document_store = ChromaDocumentStore(persist_path=self.embedding_path)
        
        # Preprocessing pipeline components
        self.file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
        self.text_file_converter = TextFileToDocument()
        self.markdown_converter = MarkdownToDocument()
        self.pdf_converter = PyPDFToDocument()
        self.document_joiner = DocumentJoiner()
        self.document_cleaner = DocumentCleaner()
        self.document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
        
        # Embedder and writer
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model=document_model, 
            progress_bar=False
        )
        self.document_embedder.warm_up()
        self.document_writer = DocumentWriter(self.document_store)
        
        # Preprocessing pipeline setup
        self.preprocessing_pipeline = Pipeline()
        self._setup_preprocessing_pipeline()
        
        # Conversation pipeline setup
        self.template = """
        You are an intelligent AI agent whose job is to understand the context from documents and whenever a user asks a question, provide answers using context from the document.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{ question }}
        Answer:
        """
        
        # Generator setup
        self.generator = OllamaGenerator(
            model=generator_model,
            url=generator_url,
            generation_kwargs={
                "num_predict": 100,
                "temperature": 0.9,
            }
        )
        
        # Conversation pipeline
        self.pipe = Pipeline()
        self._setup_conversation_pipeline(document_model)
        
        # Console for rich output
        self.console = Console()
    
    def _setup_preprocessing_pipeline(self):
        """Set up the preprocessing pipeline connections."""
        components = {
            "file_type_router": self.file_type_router,
            "text_file_converter": self.text_file_converter,
            "markdown_converter": self.markdown_converter,
            "pypdf_converter": self.pdf_converter,
            "document_joiner": self.document_joiner,
            "document_cleaner": self.document_cleaner,
            "document_splitter": self.document_splitter,
            "document_embedder": self.document_embedder,
            "document_writer": self.document_writer
        }
        
        for name, component in components.items():
            self.preprocessing_pipeline.add_component(instance=component, name=name)
        
        # Pipeline connections
        self.preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
        self.preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
        self.preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
        self.preprocessing_pipeline.connect("text_file_converter", "document_joiner")
        self.preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
        self.preprocessing_pipeline.connect("markdown_converter", "document_joiner")
        self.preprocessing_pipeline.connect("document_joiner", "document_cleaner")
        self.preprocessing_pipeline.connect("document_cleaner", "document_splitter")
        self.preprocessing_pipeline.connect("document_splitter", "document_embedder")
        self.preprocessing_pipeline.connect("document_embedder", "document_writer")
    
    def _setup_conversation_pipeline(self, document_model: str):
        """Set up the conversation pipeline connections."""
        
        self.pipe.add_component("embedder", SentenceTransformersTextEmbedder(
            model=document_model, 
            progress_bar=False
        ))
        self.pipe.add_component("retriever", ChromaEmbeddingRetriever(document_store=self.document_store, top_k=25))
        self.pipe.add_component("prompt_builder", PromptBuilder(template=self.template))
        self.pipe.add_component("llm", self.generator)
        
        self.pipe.connect("embedder.embedding", "retriever.query_embedding")
        self.pipe.connect("retriever", "prompt_builder.documents")
        self.pipe.connect("prompt_builder", "llm")
    
    def embed_documents(self, documents_path: str = None):
        """
        Embed documents from a specified path.
        
        :param documents_path: Path to documents. If None, uses the default output_dir from initialization.
        """
        if documents_path is None:
            documents_path = "lesa/samples/documents"
        
        self.preprocessing_pipeline.run({
            "file_type_router": {"sources": list(Path(documents_path).glob("**/*"))}
        })
        self.console.print(Panel(
            Text("Documents embedded successfully", style="bold green"),
            border_style="green"
        ))
        
    def embed_document(self, document_path: str = None):
        """
        Embed documents from a specified path.
        
        :param documents_path: Path to documents. If None, uses the default output_dir from initialization.
        """
        if documents_path is None:
            documents_path = "lesa/samples/documents"
        
        self.preprocessing_pipeline.run({
            "file_type_router": {"sources": list(Path('.').glob(document_path))}
        })
        self.console.print(Panel(
            Text("Documents embedded successfully", style="bold green"),
            border_style="green"
        ))
        
    def _chat(self, pipeline: Pipeline, system_prompt: Optional[str] = None):
        """
        Start an interactive chat with the RAG pipeline.
        
        :param system_prompt: Optional system-wide context or instruction
        """
        
        self.console.print(Panel.fit(
            Text("RAG Pipeline Chat Interface", style="bold magenta"),
            border_style="blue"
        ))
        
        if system_prompt:
            self.console.print(Panel(
                Text(f"System Prompt: {system_prompt}", style="dim"),
                border_style="dim"
            ))
        
        try:
            while True:
                try:
                    user_input = Prompt.ask("[bold green]You[/bold green]", password=False)
                    
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        self.console.print("[bold yellow]Exiting chat...[/bold yellow]")
                        break
                    
                    question = f"{system_prompt + ' ' if system_prompt else ''}{user_input}"
                    
                    result = pipeline.run({
                        "retriever": {"query": question},
                        "prompt_builder": {"question": question},
                        # "llm": {"generation_kwargs": {"max_new_tokens": 350}},
                    })
                    
                    response = result['llm']['replies'][0] if result['llm']['replies'] else "No response generated."
                    
                    self.console.print(Panel(
                        Text(response, style="white"),
                        title="[bold blue]RAG Response[/bold blue]",
                        border_style="blue"
                    ))
                
                except Exception as e:
                    self.console.print(Panel(
                        Text(f"Error processing query: {str(e)}", style="bold red"),
                        border_style="red"
                    ))
        
        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Chat terminated by user.[/bold yellow]")

    
    def embed_single_document_and_chat(self, file_path: Union[str, Path], persist: bool = False, document_model: str = "sentence-transformers/all-MiniLM-L6-v2",):
        """
        Embed a single document and start a conversation.
        
        :param file_path: Path to the document
        :param persist: If True, use persistent document store, else use in-memory store
        :return: List of embedded documents
        """
                
        docu_store = InMemoryDocumentStore()
        document_writer = DocumentWriter(docu_store)
        
        # file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
        # text_file_converter = TextFileToDocument()
        # markdown_converter = MarkdownToDocument()
        # pdf_converter = PyPDFToDocument()
        # document_joiner = DocumentJoiner()
        # document_cleaner = DocumentCleaner()
        # document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
        
        # document_embedder = SentenceTransformersDocumentEmbedder(
        #     model=document_model, 
        #     progress_bar=False
        # )
        # document_embedder.warm_up()
        
        # pipe = Pipeline()
        # # Add components
        # pipe.add_component(instance=file_type_router, name="file_type_router")
        # pipe.add_component(instance=text_file_converter, name="text_file_converter")
        # pipe.add_component(instance=markdown_converter, name="markdown_converter")
        # pipe.add_component(instance=pdf_converter, name="pypdf_converter")
        # pipe.add_component(instance=document_joiner, name="document_joiner")
        # pipe.add_component(instance=document_cleaner, name="document_cleaner")
        # pipe.add_component(instance=document_splitter, name="document_splitter")
        # pipe.add_component(instance=document_embedder, name="document_embedder")
        # pipe.add_component(instance=document_writer, name="document_writer")
        
        # # Connect components
        # pipe.connect("file_type_router.text/plain", "text_file_converter.sources")
        # pipe.connect("file_type_router.application/pdf", "pypdf_converter.sources")
        # pipe.connect("file_type_router.text/markdown", "markdown_converter.sources")
        # pipe.connect("text_file_converter", "document_joiner")
        # pipe.connect("pypdf_converter", "document_joiner")
        # pipe.connect("markdown_converter", "document_joiner")
        # pipe.connect("document_joiner", "document_cleaner")
        # pipe.connect("document_cleaner", "document_splitter")
        # pipe.connect("document_splitter", "document_embedder")
        # pipe.connect("document_embedder", "document_writer")
        
        # pipe = Pipeline()
        # pipe.add_component("converter", PyPDFToDocument())
        # pipe.add_component("cleaner", DocumentCleaner())
        # pipe.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
        # pipe.add_component("writer", DocumentWriter(document_store=docu_store))
        # pipe.connect("converter", "cleaner")
        # pipe.connect("cleaner", "splitter")
        # pipe.connect("splitter", "writer")
        
        converter = PyPDFToDocument()
        cleaner = DocumentCleaner()
        splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)
        writer = DocumentWriter(document_store=docu_store, policy=DuplicatePolicy.SKIP)

        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("converter", converter)
        indexing_pipeline.add_component("cleaner", cleaner)
        indexing_pipeline.add_component("splitter", splitter)
        indexing_pipeline.add_component("writer", writer)

        indexing_pipeline.connect("converter", "cleaner")
        indexing_pipeline.connect("cleaner", "splitter")
        indexing_pipeline.connect("splitter", "writer")

        
        # Run the pipeline
        indexing_pipeline.run({
            "file_type_router": {"sources": list(Path('.').glob(file_path))}
        })
        
        self.console.print(Panel(
            Text(f"Document embedded {'and persisted' if persist else 'in memory'}", style="bold green"),
            border_style="green"
        ))
        
        template = """
You are an intelligent AI agent whose job is to understand the context from documents and whenever a user asks a question, provide answers
to them using context from the document.

Context:

{% for document in documents %}
    {{ document.content }}
{% endfor %}

Please answer the question based on the given information from the given document.

{{question}}
"""
        
        generator = OllamaGenerator(model="qwen:4b",
                            url = "http://localhost:11434",
                            generation_kwargs={
                              "num_predict": 100,
                              "temperature": 0.9,
                              })
        
        single_document_chat_pipeline = Pipeline()
        single_document_chat_pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=docu_store, top_k=15))
        single_document_chat_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
        # single_document_chat_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False))
        single_document_chat_pipeline.add_component("llm", generator)
        
        # single_document_chat_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        single_document_chat_pipeline.connect("retriever", "prompt_builder.documents")
        single_document_chat_pipeline.connect("prompt_builder", "llm")
        
        self._chat(single_document_chat_pipeline)
    
    def start_conversation(self):
        """
        Start a conversation with the RAG pipeline.
        """
        self._chat(self.pipe)