import os
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from typing import List, Optional
import uvicorn
import secrets

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set")

security = HTTPBasic()
VALID_USERNAME = "demo"
VALID_PASSWORD = "legalai123"

app = FastAPI(title="Legal AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_processor = DocumentProcessor()
vector_store = VectorStore()
rag_engine = RAGEngine(vector_store)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, VALID_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, VALID_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials.username

@app.get("/", response_class=HTMLResponse)
def serve_root(request: Request):
    auth = request.cookies.get("auth")
    if auth == "true":
        return RedirectResponse(url="/index.html")
    return FileResponse("frontend/login.html")

@app.post("/login")
async def login_post(username: str = Form(...), password: str = Form(...)):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        response = RedirectResponse(url="/index.html", status_code=302)
        response.set_cookie(key="auth", value="true", httponly=True)
        return response
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/logout")
async def logout():
    response = RedirectResponse("/login.html?msg=loggedout", status_code=302)
    response.delete_cookie("auth")
    return response

@app.get("/index.html")
async def secure_index(request: Request):
    if request.cookies.get("auth") == "true":
        return FileResponse("frontend/index.html")
    return RedirectResponse("/login.html")

@app.get("/login.html")
def serve_login():
    return FileResponse("frontend/login.html")

# If you need to serve static assets like styles.css or JS, define them below as needed:
# @app.get("/assets/styles.css")
# def serve_styles():
#     return FileResponse("frontend/assets/styles.css")

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), title: str = Form(None)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        file_path, doc_id = document_processor.save_uploaded_file(file)
        doc_info = document_processor.process_document(file_path, doc_id, title)
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
    return document_processor.get_document_metadata()

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        vector_store.remove_document(doc_id)
        document_processor.remove_document(doc_id)
        return {"message": f"Document {doc_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
