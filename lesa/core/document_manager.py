import os
from abc import ABC, abstractmethod
from typing import List, Optional

import docx
import PyPDF2
# import textract

class DocumentExtractor(ABC):
    """Abstract base class for document text extraction."""
    
    @abstractmethod
    def extract_text(self, filepath: str) -> str:
        """
        Abstract method to extract text from a document.
        
        Args:
            filepath (str): Path to the document file
        
        Returns:
            str: Extracted text from the document
        """
        pass

class DocxExtractor(DocumentExtractor):
    """Extractor for .docx files using python-docx."""
    
    def extract_text(self, filepath: str) -> str:
        """
        Extract text from a Word document.
        
        Args:
            filepath (str): Path to the .docx file
        
        Returns:
            str: Extracted text from the document
        """
        try:
            doc = docx.Document(filepath)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            return f"Error extracting text from {filepath}: {str(e)}"

class PDFExtractor(DocumentExtractor):
    """Extractor for PDF files using PyPDF2."""
    
    def extract_text(self, filepath: str) -> str:
        """
        Extract text from a PDF document.
        
        Args:
            filepath (str): Path to the PDF file
        
        Returns:
            str: Extracted text from the document
        """
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return '\n'.join([page.extract_text() for page in reader.pages])
        except Exception as e:
            return f"Error extracting text from {filepath}: {str(e)}"

# class GenericTextExtractor(DocumentExtractor):
#     """Extractor for generic text-based files using textract."""
    
#     def extract_text(self, filepath: str) -> str:
#         """
#         Extract text from various text-based document formats.
        
#         Args:
#             filepath (str): Path to the document file
        
#         Returns:
#             str: Extracted text from the document
#         """
#         try:
#             return textract.process(filepath).decode('utf-8')
#         except Exception as e:
#             return f"Error extracting text from {filepath}: {str(e)}"

class DocumentManager:
    """
    Manages document selection and extraction.
    
    Responsibilities:
    - Scan directory for supported documents
    - Provide document selection interface
    """
    
    def __init__(self, directory: str = '.'):
        """
        Initialize DocumentManager.
        
        Args:
            directory (str, optional): Directory to scan. Defaults to current directory.
        """
        self.directory = directory
        self.supported_extensions = {
            '.docx': DocxExtractor(),
            '.pdf': PDFExtractor(),
            # '.txt': GenericTextExtractor(),
            # '.rtf': GenericTextExtractor(),
            # '.odt': GenericTextExtractor()
        }
        self.documents: List[str] = []
    
    def scan_documents(self) -> List[str]:
        """
        Scan the directory for supported document types.
        
        Returns:
            List[str]: List of supported document filenames
        """
        self.documents = []
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if os.path.isfile(filepath):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in self.supported_extensions:
                    self.documents.append(filepath)
        
        return self.documents
    
    def list_documents(self) -> None:
        """
        Display the list of available documents with indices.
        """
        if not self.documents:
            print("No supported documents found.")
            return
        
        print("Available documents:")
        for idx, doc in enumerate(self.documents, 1):
            print(f"{idx}. {os.path.basename(doc)}")
    
    def select_document(self) -> Optional[str]:
        """
        Interactively select a document to extract text from.
        
        Returns:
            Optional[str]: Path to the selected document, or None if no selection
        """
        # Ensure documents are scanned
        if not self.documents:
            self.scan_documents()
        
        # List documents
        self.list_documents()
        
        # No documents found
        if not self.documents:
            return None
        
        # User selection loop
        while True:
            try:
                selection = input("\nEnter the number of the document to view (or 'q' to quit): ")
                
                # Exit condition
                if selection.lower() == 'q':
                    return None
                
                # Validate selection
                idx = int(selection)
                if 1 <= idx <= len(self.documents):
                    return self.documents[idx - 1]
                else:
                    print("Invalid selection. Please try again.")
            
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
    
    def extract_document_text(self, filepath: str) -> str:
        """
        Extract text from a document using the appropriate extractor.
        
        Args:
            filepath (str): Path to the document
        
        Returns:
            str: Extracted text from the document
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Select appropriate extractor
        extractor = self.supported_extensions.get(file_ext)
        
        if extractor:
            return extractor.extract_text(filepath)
        else:
            return f"No extractor found for file type: {file_ext}"