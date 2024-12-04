import argparse
import os
import sys
from typing import List

# Haystack and related imports
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import Document

from lesa.core.document_manager import DocumentManager

dm = DocumentManager()

# LLM interaction
from haystack_integrations.components.generators.ollama import OllamaGenerator

def load_document(file_path: str) -> List[Document]:
    """
    Load and convert document to Haystack Documents.
    
    :param file_path: Path to the document
    :return: List of Haystack Documents
    """
    # reader = get_document_reader(file_path)
    
    # if reader:
    #     # Use reader for PDFs, DOCXs
    #     docs = reader.read(file_path)
    # else:
    #     # For text files, read directly
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         text = f.read()
    #         docs = [Document(text)]
    
    docs = [Document(content=dm.extract_document_text(file_path))]
    
    return docs

def create_rag_pipeline(document_store, model_name: str = "mistral"):
    """
    Create a Retrieval-Augmented Generation (RAG) pipeline.
    
    :param document_store: ChromaDB document store
    :param model_name: Ollama model name
    :return: Configured Haystack pipeline
    """
    # Document embedder to create embeddings for documents
    doc_embedder = SentenceTransformersDocumentEmbedder()
    
    # Text embedder for queries
    text_embedder = SentenceTransformersTextEmbedder()
    
    # Retriever to find relevant documents
    retriever = ChromaEmbeddingRetriever(document_store=document_store)
    
    # Prompt builder for RAG
    prompt_template = """
    Context: {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ query }}

    Based on the context, please provide a detailed and helpful answer.
    """
    prompt_builder = PromptBuilder(template=prompt_template)
    
    # Ollama generator
    generator = OllamaGenerator(model=model_name)
    
    # Create pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", generator)
    
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "generator")
    
    return rag_pipeline

def main():
    print("JHELLO")
    # Argument parsing
    parser = argparse.ArgumentParser(description="RAG CLI Tool with Haystack, ChromaDB, and Ollama")
    parser.add_argument("document", help="Path to the document to be processed")
    parser.add_argument("--model", default="llama3", help="Ollama model name (default: llama3)")
    parser.add_argument("--max-docs", type=int, default=5, help="Maximum number of retrieved documents")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.document):
        print(f"Error: Document {args.document} not found.")
        sys.exit(1)
    
    # Load document
    print(f"Loading document: {args.document}")
    documents = load_document(args.document)
    
    # Create document store
    document_store = ChromaDocumentStore()
    
    # Preprocess documents
    # preprocessor = TextPreprocessor(split_length=200, split_overlap=20)
    splitter = DocumentSplitter(split_by="passage", split_length=200, split_overlap=20)
    split_docs = splitter.run(documents)
    preprocessor = DocumentCleaner()
    preprocessed_docs = preprocessor.run(split_docs['documents'])
    
    # Create document embeddings and store
    doc_embedder = SentenceTransformersDocumentEmbedder()
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(preprocessed_docs['documents'])
    document_store.write_documents(docs_with_embeddings['documents'])
    
    # Create RAG pipeline
    rag_pipeline = create_rag_pipeline(document_store, args.model)
    
    # Interactive chat loop
    print(f"\n🤖 RAG Chat initialized with {args.model} model. Type 'exit' to quit.")
    print("------------------------------------------------")
    
    while True:
        try:
            query = input("\n> ")
            
            if query.lower() == 'exit':
                break
            
            # Run pipeline
            result = rag_pipeline.run({
                "text_embedder": {"text": query},
                "retriever": {"top_k": args.max_docs},
                "prompt_builder": {"query": query}
            })
            
            # Print response
            print("\n🔍 Response:", result['generator']['replies'][0])
        
        except KeyboardInterrupt:
            print("\n\nChat terminated by user.")
            break
    
    print("\nThank you for using the RAG CLI Tool!")

if __name__ == "__main__":
    main()

# Requirements:
# pip install haystack-ai
# pip install haystack-ai-integrations
# pip install ollama
# pip install sentence-transformers
# pip install python-docx
# pip install chromadb