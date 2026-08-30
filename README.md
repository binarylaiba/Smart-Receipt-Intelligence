---
title: AI Receipt Analyser & Financial Assistant
emoji: 🧾
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🧾 AI Receipt Analyser & Financial Assistant

An end-to-end AI system that scans receipts using **Computer Vision (OpenCV)**, extracts raw text with **EasyOCR**, structures items into a validated financial contract with **Groq LLM (Llama 3.3 70B)** + **Pydantic v2**, and generates spending analytics with **Pandas & Plotly**.

---

## 📁 Project Structure

```text
AI RECIPT ANALYSER/
│
├── .env                  # Environment configuration (GROQ_API_KEY)
├── requirements.txt      # Production package dependencies
├── packages.txt          # Linux system dependencies for Streamlit Cloud
├── README.md             # Project documentation & run guide
│
├── ocr_engine.py         # Model 1: Image preprocessing & EasyOCR text extraction
├── llm_parser.py         # Model 2: Groq LLM contract structuring & financial advice
├── analytics.py          # Pandas financial analytics & arithmetic audit checks
│
├── app.py                # Full-Stack Streamlit & Plotly Interactive Dashboard
├── api.py                # Optional FastAPI REST backend server
└── test_client.py        # Sample API request testing script
```

---

## 🚀 Quickstart Guide

### 1. Set Up Virtual Environment

**Windows (CMD / PowerShell):**
```cmd
py -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Configure API Key

Create or update `.env` in the project root:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

*(Or set directly in terminal session)*:
- **Windows CMD:** `set GROQ_API_KEY=gsk_your_groq_api_key_here`
- **Windows PowerShell:** `$env:GROQ_API_KEY="gsk_your_groq_api_key_here"`
- **macOS / Linux:** `export GROQ_API_KEY="gsk_your_groq_api_key_here"`

---

### 4. Launch the Application

#### Option A: Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
*Access in browser at:* `http://localhost:8501`

#### Option B: FastAPI Backend Server
```bash
py api.py
```
*Interactive Swagger docs at:* `http://127.0.0.1:8000/docs`

---

## ☁️ Streamlit Community Cloud Deployment (Secrets Setup)

When deploying this repository to **Streamlit Community Cloud** (`share.streamlit.io`):
1. Note that `.env` files are ignored by git for security reasons and will NOT be on GitHub.
2. In your Streamlit Cloud Dashboard, go to your deployed app.
3. Click **App Settings** (or the `⋮` / settings menu at bottom right) ➜ **Secrets**.
4. Add your Groq API key:
```toml
GROQ_API_KEY = "gsk_your_groq_api_key_here"
```
5. Click **Save** and restart the app.
6. Alternatively, you or any user can enter the Groq API key directly into the sidebar text input inside the app UI.

---

## 🛠️ Troubleshooting Common Issues

### 1. EasyOCR Model Download / PyTorch
* **Issue:** EasyOCR takes extra time on the first run.
* **Fix:** EasyOCR automatically downloads detection/recognition neural network weights (`~100MB`) to `~/.EasyOCR/` on the initial execution.
* **CPU vs GPU:** If you do not have CUDA installed, EasyOCR automatically runs on CPU without any extra setup.

### 2. OpenCV Headless vs GUI Conflict
* **Issue:** `ImportError: cannot import name '...' from 'cv2'` or Qt platform plugin errors.
* **Fix:** Use `opencv-python-headless` instead of `opencv-python` to avoid desktop GUI dependencies in server/Streamlit environments.

### 3. Groq Model Fallbacks
* **Active Models:** `llm_parser.py` includes automatic fallback across active high-speed models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `llama3-70b-8192`).

---

## 📜 License
MIT License - Open for personal and commercial usage.
