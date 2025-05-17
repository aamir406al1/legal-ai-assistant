import os
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from typing import List, Optional
import uvicorn

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set")

# Initialize the FastAPI app
app = FastAPI(title="Legal AI Assistant API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor()
vector_store = VectorStore()
rag_engine = RAGEngine(vector_store)

# Endpoints
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(None)
):
    """Upload and process a legal document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save and process the uploaded file
        file_path, doc_id = document_processor.save_uploaded_file(file)
        doc_info = document_processor.process_document(file_path, doc_id, title)
        
        # Create and store embeddings
        embeddings = document_processor.create_embeddings(doc_info["chunks"])
        vector_store.add_document(doc_id, doc_info["chunks"], embeddings)
        
        return {
            "message": "Document uploaded and processed successfully",
            "doc_id": doc_id,
            "title": doc_info["title"],
            "chunks": doc_info["total_chunks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return document_processor.get_document_metadata()

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the system"""
    try:
        vector_store.remove_document(doc_id)
        document_processor.remove_document(doc_id)
        return {"message": f"Document {doc_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query")
# async def query_documents(
#     query: str,
#     doc_filter: Optional[List[str]] = None
# ):
#     """Query the legal documents"""
#     try:
#         response = rag_engine.query(query, doc_filter)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_endpoint(request: Request):
    try:
        body = await request.json()
        print("📥 Received body:", body)
        query_text = body.get("query")
        doc_filter = body.get("doc_filter")

        result = rag_engine.query(query_text=query_text, doc_filter=doc_filter)
        return result
    except Exception as e:
        print("❌ Backend Exception:", e)
        return {"error": str(e)}


# Mount the static files directory for the frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)