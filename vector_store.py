import os
import pickle
import numpy as np
from typing import List, Dict, Any
import faiss

class VectorStore:
    def __init__(self, vector_db_path="vector_db"):
        self.vector_db_path = vector_db_path
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Index of document chunks
        self.chunks_index_file = os.path.join(vector_db_path, "chunks_index.pkl")
        self.chunks_index = {}
        
        # FAISS indices by document
        self.faiss_indices = {}
        
        # Load existing indices
        self._load_indices()
    
    def _load_indices(self):
        """Load existing indices from disk"""
        if os.path.exists(self.chunks_index_file):
            with open(self.chunks_index_file, "rb") as f:
                self.chunks_index = pickle.load(f)
        
        # Load FAISS indices for each document
        for doc_id in self.chunks_index:
            index_file = os.path.join(self.vector_db_path, f"{doc_id}.index")
            if os.path.exists(index_file):
                self.faiss_indices[doc_id] = faiss.read_index(index_file)
    
    def add_document(self, doc_id: str, chunks: List, embeddings: List[List[float]]):
        """Add a document's chunks and embeddings to the vector store"""
        # Store document chunks
        self.chunks_index[doc_id] = chunks
        
        # Create and store FAISS index
        embeddings_array = np.array(embeddings).astype("float32")
        dimension = embeddings_array.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Save the index
        self.faiss_indices[doc_id] = index
        faiss.write_index(index, os.path.join(self.vector_db_path, f"{doc_id}.index"))
        
        # Save updated chunks index
        with open(self.chunks_index_file, "wb") as f:
            pickle.dump(self.chunks_index, f)
    
    def search(self, query_embedding: List[float], top_k: int = 5, doc_filter: List[str] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks across all documents or filtered documents"""
        results = []
        query_embedding_array = np.array([query_embedding]).astype("float32")
        
        # Filter documents if specified
        doc_ids = doc_filter if doc_filter else list(self.chunks_index.keys())
        
        for doc_id in doc_ids:
            if doc_id not in self.faiss_indices:
                continue
                
            index = self.faiss_indices[doc_id]
            distances, indices = index.search(query_embedding_array, min(top_k, index.ntotal))
            
            doc_chunks = self.chunks_index[doc_id]
            
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    chunk = doc_chunks[idx]
                    results.append({
                        "chunk": chunk,
                        "distance": float(distances[0][i]),
                        "doc_id": doc_id,
                        "chunk_id": idx
                    })
        
        # Sort by distance (lower is better)
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]
    
    def remove_document(self, doc_id: str):
        """Remove a document from the vector store"""
        if doc_id in self.chunks_index:
            del self.chunks_index[doc_id]
        
        if doc_id in self.faiss_indices:
            del self.faiss_indices[doc_id]
        
        # Remove the index file
        index_file = os.path.join(self.vector_db_path, f"{doc_id}.index")
        if os.path.exists(index_file):
            os.remove(index_file)
        
        # Save updated chunks index
        with open(self.chunks_index_file, "wb") as f:
            pickle.dump(self.chunks_index, f)
    
    def list_documents(self):
        """List all documents in the vector store"""
        return list(self.chunks_index.keys())