from pathlib import Path
import os
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from typing import Optional, List, Union
import traceback

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain.chains.base import Chain

from lesa.core.ollama import OllamaManager
from lesa.core.directory_manager import DirectoryManager
from lesa.utils.pre_prompt import pre_prompt


class ConversationManager:

    CONFIG_DIR = ".lesa"

    def __init__(
        self,
        base_path: str = ".",
        document_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the RAG pipeline with configurable parameters.

        :param output_dir: Directory containing source documents
        :param embedding_path: Path to store embeddings
        :param document_model: Embedding model for documents
        :param generator_model: LLM model for generating responses
        :param generator_url: URL for the LLM generator
        """

        self.base_path = os.path.abspath(base_path)
        self.embedding_path = os.path.join(
            self.base_path, self.CONFIG_DIR, "embeddings"
        )
        self.ollama_manager = OllamaManager()
        self.embeddings_manager = DirectoryManager(document_model=document_model)

        # Console for rich output
        self.console = Console()

    def _setup_preprocessing_pipeline(self):
        """Set up the preprocessing pipeline connections."""
        pass

    def _setup_conversation_pipeline(self, document_model: str):
        """Set up the conversation pipeline connections."""

        pass

    def _chat(
        self,
        chain: Chain,
        system_prompt: Optional[str] = None,
        stream: bool = True,
        context=Optional[list[Document]],
    ):
        """
        Start an interactive chat with the RAG pipeline.

        :param system_prompt: Optional system-wide context or instruction
        :param context: Optional list of documents to use as context
        """

        self.console.print(
            Panel(
                Text(
                    "📚 Turn your terminal into a File Interpreter",
                    style="bold green",
                    justify="center",
                ),
                border_style="green",
                title="Lesa",
            )
        )

        if system_prompt:
            self.console.print(
                Panel(
                    Text(f"System Prompt: {system_prompt}", style="dim"),
                    border_style="dim",
                )
            )

        try:
            while True:
                try:
                    user_input = Prompt.ask(
                        "[bold green]You[/bold green]", password=False
                    )

                    if user_input.lower() in ["exit", "quit", "q"]:
                        self.console.print("[bold yellow]Exiting chat...[/bold yellow]")
                        break

                    question = (
                        f"{system_prompt + ' ' if system_prompt else ''}{user_input}"
                    )

                    if stream:
                        # Stream response (Faster)
                        self.console.print(Text("Lesa: ", style="bold deep_pink1"), end="")
                        if context:
                            for token in chain.stream(
                                {"input": question, "context": context}
                            ):
                                self.console.print(token, end="")
                            self.console.print()
                        else:
                            for token in chain.stream({"input": question}):
                                self.console.print(token, end="")
                            self.console.print()
                        time.sleep(1)
                    
                    else:
                        # Complete Response at once (Takes longer)
                        with self.console.status("🧠 Thinking...") as status:
                            result = None
                            if context:
                                result = chain.invoke({"input": question, "context": context})
                            else:
                                result = chain.invoke({"input": question})
                            status.update("🎉 Done!")
                            time.sleep(1)

                        try:
                            # Llama3 returns a dictionary with 'answer' key which contains the response
                            response = result['answer'] if result['answer'] else "No response generated."
                            self.console.print(
                                Text("Lesa: ", style="bold deep_pink1") + Text(response, style="bold white")
                            )
                        except Exception as e:
                            # Qwen returns a string response
                            self.console.print(
                                Text("Lesa: ", style="bold deep_pink1") + Text(result, style="bold white")
                            )

                except Exception as e:
                    self.console.print(
                        Panel(
                            Text(f"Error processing query: {e}", style="bold red")
                            + Text(traceback.format_exc(), style="bold yellow"),
                            border_style="red",
                        )
                    )

        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Chat terminated by user.[/bold yellow]")

    def embed_single_document_and_chat(
        self,
        file_path: Union[str, Path],
        persist: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Embed a single document and start a conversation.

        :param file_path: Path to the document
        :param persist: If True, use persistent document store, else use in-memory store
        :return: List of embedded documents
        """

        documents = self.embeddings_manager.extract_file_text(file_path)
        docs = self.embeddings_manager.text_splitter.split_documents(
            documents=documents
        )

        # Embed documents
        self.embeddings_manager.embed_documents(docs)

        llm = self.ollama_manager.serve_llm()

        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    pre_prompt()
                    + """Answer any use questions based solely on the context below:

<context>
{context}
</context>.
You can use your general knowledge to supplement the context provided.""",
                ),
                ("human", """{input}"""),
            ]
        )

        combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        qa_chain = create_retrieval_chain(
            retriever=self.embeddings_manager.vector_store.as_retriever(),
            combine_docs_chain=combine_docs_chain,
        )
        return self._chat(qa_chain)

    def start_conversation(self):
        """
        Start a conversation with the RAG pipeline.
        """

        llm = self.ollama_manager.serve_llm()
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    pre_prompt()
                    + """
Answer any use questions based solely on the context below:

<context>
{context}
</context>""",
                ),
                ("human", """{input}"""),
            ]
        )

        combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        qa_chain = create_retrieval_chain(
            retriever=self.embeddings_manager.vector_store.as_retriever(),
            combine_docs_chain=combine_docs_chain,
        )
        return self._chat(qa_chain)

    def single_page_chat(
        self,
        file_path: Union[str, Path],
        page_number: Optional[int] = None,
    ):
        """"""

        docs = self.embeddings_manager.extract_file_text(
            filepath=file_path, page_number=page_number
        )
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    pre_prompt()
                    + """Answer any use questions based solely on the context given:\n\n\n{context}\n
You can use your general knowledge to supplement the context provided.""",
                ),
                ("human", """{input}"""),
            ]
        )

        chain = create_stuff_documents_chain(
            llm=self.ollama_manager.serve_llm(), prompt=prompt
        )
        return self._chat(chain, context=docs)
