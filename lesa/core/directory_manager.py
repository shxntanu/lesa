import os
import json
import hashlib
from typing import Dict, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from rich.console import Console
console = Console()

import faiss
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

class DocumentExtractor(ABC):
    """Abstract base class for document text extraction."""
    
    @abstractmethod
    def extract_text(self, filepath: str) -> list[Document]:
        """
        Abstract method to extract text from a document.
        
        Args:
            filepath (str): Path to the document file
        
        Returns:
            list[Document]: List of extracted documents
        """
        pass

class DocxExtractor(DocumentExtractor):
    """Extractor for .docx files using python-docx."""
    
    def extract_text(self, filepath: str) -> list[Document]:
        """
        Extract text from a Word document.
        
        Args:
            filepath (str): Path to the .docx file
        
        Returns:
            list[Document]: List of extracted documents
        """
        try:
            loader = Docx2txtLoader(file_path=filepath)
            documents = loader.load()
            return documents
        except Exception as e:
            return f"Error extracting text from {filepath}: {str(e)}"

class PDFExtractor(DocumentExtractor):
    """Extractor for PDF files using PyPDF2."""
    
    def extract_text(self, filepath: str) -> list[Document]:
        """
        Extract text from a PDF document.
        
        Args:
            filepath (str): Path to the PDF file
        
        Returns:
            str: Extracted text from the document
        """
        try:
            loader = PyPDFLoader(file_path=filepath)
            documents = loader.load()
            return documents
        except Exception as e:
            return f"Error extracting text from {filepath}: {str(e)}"

class DirectoryManager:
    """
    Configuration manager for Lesa to track file states and changes.
    """
    
    CONFIG_DIR = '.lesa'
    CONFIG_FILE = 'config.json'
    
    def __init__(self, 
                 base_path: str = '.',
                 document_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 console: Console = None
        ):
        """
        Initialize the configuration manager for a specific directory.
        
        :param base_path: Base directory to manage (default is current directory)
        """
        self.base_path = os.path.abspath(base_path)
        self.config_path = os.path.join(self.base_path, self.CONFIG_DIR)
        self.config_file_path = os.path.join(self.config_path, self.CONFIG_FILE)
        self.files: List[str] = []
        self.directory = base_path
        self.supported_extensions = {
            '.docx': DocxExtractor(),
            '.pdf': PDFExtractor(),
        }
        
        self.document_model = document_model
        self.embeddings = HuggingFaceEmbeddings(model_name=document_model)
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        
        # Load the vector store if it exists
        if os.path.isdir(os.path.join(os.getcwd(), '.lesa/embeddings')):
            self.vector_store = FAISS.load_local(
                folder_path=os.path.join(os.getcwd(), '.lesa/embeddings'),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS(
                embedding_function=self.embeddings, 
                docstore=InMemoryDocstore(),
                index=faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world"))),
                index_to_docstore_id={}
            )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        :param file_path: Path to the file
        :return: File hash as a string
        """
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_file_metadata(self, file_path: str) -> Dict:
        """
        Get metadata for a specific file.
        
        :param file_path: Path to the file
        :return: Dictionary with file metadata
        """
        stat = os.stat(file_path)
        return {
            'path': os.path.relpath(file_path, self.base_path),
            'hash': self._calculate_file_hash(file_path),
            'size': stat.st_size,
            'modified_time': stat.st_mtime
        }
    
    def init(self) -> None:
        """
        Initialize the Lesa configuration for the current directory.
        Creates config directory and initial configuration snapshot.
        """
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_path, exist_ok=True)
        
        file_metadata = self._scan_directory()
        
        config_data = {
            'initialized_at': datetime.now().isoformat(),
            'base_path': self.base_path,
            'files': file_metadata
        }
        
        with open(self.config_file_path, 'w') as f:
            json.dump(config_data, f, indent=4)
            
        self.scan_files()
        # print(f"Lesa embeddings initialized in {self.base_path}")
    
    def _scan_directory(self, ignore_patterns: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Scan the directory and collect metadata for all files.
        
        :param ignore_patterns: List of patterns to ignore (e.g., ['.lesa', '.git'])
        :return: Dictionary of file metadata
        """
        ignore_patterns = ignore_patterns or [self.CONFIG_DIR, '.git']
        file_metadata = {}
        
        for root, dirs, files in os.walk(self.base_path):
            dirs[:] = [d for d in dirs if not any(ig in d for ig in ignore_patterns)]
            
            for file in files:
                full_path = os.path.join(root, file)
                
                if not any(ig in full_path for ig in ignore_patterns):
                    try:
                        metadata = self._get_file_metadata(full_path)
                        file_metadata[metadata['path']] = metadata
                    except Exception as e:
                        print(f"Could not process file {full_path}: {e}")
        
        return file_metadata
    
    def check_for_changes(self) -> Dict:
        """
        Check if any files have changed since last initialization.
        
        :return: Dictionary of change types
        """
        # Ensure config file exists
        if not os.path.exists(self.config_file_path):
            raise FileNotFoundError("Lesa embeddings not initialized. Run 'lesa init' first.")
        
        # Read existing configuration
        with open(self.config_file_path, 'r') as f:
            existing_config = json.load(f)
        
        current_files = self._scan_directory()
        
        changes = {
            'new_files': [],
            'deleted_files': [],
            'modified_files': []
        }
        
        # Check for new and modified files
        for path, metadata in current_files.items():
            if path not in existing_config['files']:
                changes['new_files'].append(path)
            elif (existing_config['files'][path]['hash'] != metadata['hash'] or 
                  existing_config['files'][path]['size'] != metadata['size']):
                changes['modified_files'].append(path)
        
        # Check for deleted files
        for path in existing_config['files']:
            if path not in current_files:
                changes['deleted_files'].append(path)
        
        return changes
    
    def update_configuration(self) -> None:
        """
        Update the configuration after detecting changes.
        Rescan the directory and update the configuration file.
        """
        file_metadata = self._scan_directory()
        
        config_data = {
            'initialized_at': datetime.now().isoformat(),
            'base_path': self.base_path,
            'files': file_metadata
        }
        
        with open(self.config_file_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        console.print("Configuration updated successfully")
        
    def scan_files(self) -> List[str]:
        """
        Scan the directory for supported document types.
        
        Returns:
            List[str]: List of supported document filenames
        """
        self.files = []
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if os.path.isfile(filepath):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in self.supported_extensions:
                    self.files.append(filepath)
        
        return self.files
    
    def list_files(self) -> None:
        """
        Display the list of available files with indices.
        """
        if not self.files:
            print("No supported files found.")
            return
        
        print("Available files:")
        for idx, doc in enumerate(self.files, 1):
            print(f"{idx}. {os.path.basename(doc)}")
            
    def extract_file_text(self, filepath: str) -> list[Document]:
        """
        Extract text from a document using the appropriate extractor.
        
        Args:
            filepath (str): Path to the document
        
        Returns:
            list[Document]: List of extracted documents
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Select appropriate extractor
        extractor = self.supported_extensions.get(file_ext)
        if extractor:
            documents = extractor.extract_text(filepath)
            return self.text_splitter.split_documents(documents=documents)
        else:
            return f"No extractor found for file type: {file_ext}"
        
    def embed_documents(self, documents: list[Document]):
        """
        Embed a list of documents into the vector store.
        
        :param documents: List of documents to embed
        :return: List of document embeddings
        """
        
        try: 
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error embedding documents: {e}")
            return False
        
    def save_vector_store(self):
        """
        Save the vector store to disk.
        """
        
        try:
            self.vector_store.save_local(os.path.join(os.getcwd(), '.lesa/embeddings'))
            return True
        except Exception as e:
            print(f"Error saving vector store: {e}")
            return False