from pathlib import Path

output_dir = "lesa/samples/documents"

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from typing import Optional

from haystack.components.writers import DocumentWriter
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

document_store = ChromaDocumentStore(persist_path="lesa/samples/embeddings")
file_type_router = FileTypeRouter(
    mime_types=["text/plain", "application/pdf", "text/markdown"]
)
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
pdf_converter = PyPDFToDocument()
document_joiner = DocumentJoiner()

document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(
    split_by="word", split_length=150, split_overlap=50
)

document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False
)
document_embedder.warm_up()
document_writer = DocumentWriter(document_store)

preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
preprocessing_pipeline.add_component(
    instance=text_file_converter, name="text_file_converter"
)
preprocessing_pipeline.add_component(
    instance=markdown_converter, name="markdown_converter"
)
preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
preprocessing_pipeline.add_component(
    instance=document_splitter, name="document_splitter"
)
preprocessing_pipeline.add_component(
    instance=document_embedder, name="document_embedder"
)
preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

preprocessing_pipeline.connect(
    "file_type_router.text/plain", "text_file_converter.sources"
)
preprocessing_pipeline.connect(
    "file_type_router.application/pdf", "pypdf_converter.sources"
)
preprocessing_pipeline.connect(
    "file_type_router.text/markdown", "markdown_converter.sources"
)
preprocessing_pipeline.connect("text_file_converter", "document_joiner")
preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
preprocessing_pipeline.connect("markdown_converter", "document_joiner")
preprocessing_pipeline.connect("document_joiner", "document_cleaner")
preprocessing_pipeline.connect("document_cleaner", "document_splitter")
preprocessing_pipeline.connect("document_splitter", "document_embedder")
preprocessing_pipeline.connect("document_embedder", "document_writer")

preprocessing_pipeline.run(
    {"file_type_router": {"sources": list(Path(output_dir).glob("**/*"))}}
)

template = """
You are an intelligent AI agent whose job is to understand the context from documents and whenever a user asks a question, provide answers
to them using context from the document.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""

generator = OllamaGenerator(
    model="qwen:4b",
    url="http://localhost:11434",
    generation_kwargs={
        "num_predict": 100,
        "temperature": 0.9,
    },
)

pipe = Pipeline()
pipe.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False
    ),
)
pipe.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)

pipe.connect("embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

app = typer.Typer()
console = Console()


@app.command()
def chat(
    system_prompt: Optional[str] = typer.Option(
        None, help="Optional system-wide context or instruction"
    )
):
    """
    Start an interactive chat with the RAG pipeline.

    Use Ctrl+C to exit the chat.
    """
    console.print(
        Panel.fit(
            Text("RAG Pipeline Chat Interface", style="bold magenta"),
            border_style="blue",
        )
    )

    if system_prompt:
        console.print(
            Panel(
                Text(f"System Prompt: {system_prompt}", style="dim"), border_style="dim"
            )
        )

    try:
        while True:
            user_input = Prompt.ask("[bold green]You[/bold green]", password=False)

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[bold yellow]Exiting chat...[/bold yellow]")

                break

            try:
                question = f"{system_prompt + ' ' if system_prompt else ''}{user_input}"

                result = pipe.run(
                    {
                        "embedder": {"text": question},
                        "prompt_builder": {"question": question},
                        "llm": {"generation_kwargs": {"max_new_tokens": 350}},
                    }
                )

                response = (
                    result["llm"]["replies"][0]
                    if result["llm"]["replies"]
                    else "No response generated."
                )

                console.print(
                    Panel(
                        Text(response, style="white"),
                        title="[bold blue]RAG Response[/bold blue]",
                        border_style="blue",
                    )
                )

            except Exception as e:
                console.print(
                    Panel(
                        Text(f"Error processing query: {str(e)}", style="bold red"),
                        border_style="red",
                    )
                )

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Chat terminated by user.[/bold yellow]")


if __name__ == "__main__":
    app()
