from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid
import json
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Multimodal RAG System")

# Add CORS middleware to allow requests from the UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the RAG system
rag_system = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    modality: Optional[str] = "all"

@app.get("/")
async def root():
    """Redirect to UI."""
    return FileResponse("ui/index.html")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process PDF files for the RAG system."""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    results = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            results.append({
                "filename": file.filename,
                "status": "skipped",
                "message": "Only PDF files are supported"
            })
            continue
        
        # Create a unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join("temp", unique_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Process the document with the RAG system
            doc_id = rag_system.add_document(file_path)
            
            if doc_id:
                results.append({
                    "filename": file.filename,
                    "document_id": doc_id,
                    "status": "processed"
                })
            else:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "Processing failed"
                })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
            
            # Clean up the file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return {"results": results}

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system with a question."""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        results = rag_system.query(
            request.query, 
            modality=request.modality, 
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the system is running."""
    return {
        "status": "healthy", 
        "rag_initialized": rag_system is not None
    }