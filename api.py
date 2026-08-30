"""FastAPI Backend Server for AI Receipt Analyser.

Exposes REST endpoints to upload receipt images and receive
structured financial analyses.
"""

from __future__ import annotations

import io
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file, override=True)
else:
    load_dotenv(override=True)

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ocr_engine import preprocess_image, extract_raw_text
from llm_parser import parse_receipt_with_groq, ReceiptAnalysis

# Initialize FastAPI application
app = FastAPI(
    title="AI Receipt Analyser API",
    description="Full-pipeline OCR and LLM-powered receipt analysis service.",
    version="1.0.0",
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint to verify server status."""
    return {"status": "online", "message": "AI Receipt Analyser API is running smoothly."}


@app.post(
    "/api/analyze-receipt",
    response_model=ReceiptAnalysis,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Upload receipt image for OCR and LLM financial analysis",
)
async def analyze_receipt(
    file: UploadFile = File(..., description="Receipt image file (JPEG, PNG, WebP, etc.)"),
):
    """Receive an uploaded receipt image, preprocess it, extract text via OCR,

    and run LLM structuring with Groq.
    """
    # 1. Validate file extension/MIME
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image format (JPEG, PNG, etc.).",
        )

    try:
        # Read image bytes
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # 2. Run Computer Vision Preprocessing
        preprocessed_np, _ = preprocess_image(image_bytes)

        # 3. Run OCR Text Extraction
        raw_text = extract_raw_text(preprocessed_np)
        if not raw_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not detect any readable text on the uploaded receipt.",
            )

        # 4. Run LLM Structured Analysis via Groq
        structured_data = parse_receipt_with_groq(raw_text)

        return structured_data

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process receipt: {str(exc)}",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    # Start server locally on port 8000
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
