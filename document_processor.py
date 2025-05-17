import os
import uuid
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import pandas as pd

class DocumentProcessor:
    def __init__(self, documents_dir="documents", embeddings_model=None):
        self.documents_dir = documents_dir
        self.metadata_file = "document_metadata.csv"
        os.makedirs(documents_dir, exist_ok=True)
        
        # Initialize embeddings model
        self.embeddings_model = embeddings_model or OpenAIEmbeddings()
        
        # Text splitter configuration optimized for legal documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        # Initialize or load document metadata
        if os.path.exists(self.metadata_file):
            self.document_metadata = pd.read_csv(self.metadata_file)
        else:
            self.document_metadata = pd.DataFrame(columns=[
                "doc_id", "filename", "title", "chunks", "upload_date"
            ])
            self.document_metadata.to_csv(self.metadata_file, index=False)
    
    def save_uploaded_file(self, file) -> str:
        """Save an uploaded file and return the path"""
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(self.documents_dir, f"{doc_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        return file_path, doc_id
    
    def process_document(self, file_path: str, doc_id: str, title: str = None) -> Dict[str, Any]:
        """Process a document file into chunks with metadata"""
        title = title or os.path.basename(file_path).split('_', 1)[1]
        
        # Extract text from PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Add page numbers and document info to metadata
        for i, page in enumerate(pages):
            page.metadata.update({
                "doc_id": doc_id,
                "title": title,
                "page": i + 1,
                "source": file_path
            })
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        
        # Update document metadata
        new_row = {
            "doc_id": doc_id,
            "filename": os.path.basename(file_path),
            "title": title,
            "chunks": len(chunks),
            "upload_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.document_metadata = pd.concat([
            self.document_metadata, 
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        self.document_metadata.to_csv(self.metadata_file, index=False)
        
        return {
            "doc_id": doc_id,
            "title": title,
            "chunks": chunks,
            "total_chunks": len(chunks),
        }
    
    def create_embeddings(self, chunks: List) -> List[List[float]]:
        """Create embeddings for document chunks"""
        texts = [chunk.page_content for chunk in chunks]
        return self.embeddings_model.embed_documents(texts)
    
    def remove_document(self, doc_id: str):
        """Remove document metadata and the associated PDF file"""
        # Find the row for the doc_id
        row = self.document_metadata[self.document_metadata["doc_id"] == doc_id]
        if not row.empty:
            filename = row.iloc[0]["filename"]
            file_path = os.path.join(self.documents_dir, filename)
            
            # Delete the file
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove metadata
        self.document_metadata = self.document_metadata[self.document_metadata["doc_id"] != doc_id]
        self.document_metadata.to_csv(self.metadata_file, index=False)

    
    def get_document_metadata(self):
        """Return all document metadata"""
        if os.path.exists(self.metadata_file):
            return pd.read_csv(self.metadata_file).to_dict(orient="records")
        return []