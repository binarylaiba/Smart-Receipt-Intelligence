# 🧾 AI Receipt Analyser & Financial Assistant

An end-to-end AI system that scans receipts using **Computer Vision (OpenCV)**, extracts raw text with **EasyOCR**, structures items into a validated financial contract with **Groq LLM (Llama / GPT-OSS)** + **Pydantic v2**, and generates spending analytics with **Pandas & Plotly**.

---

## 📁 Project Structure

```text
AI RECIPT ANALYSER/
│
├── .env                  # Environment configuration (GROQ_API_KEY)
├── requirements.txt      # Production package dependencies
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

## 🛠️ Troubleshooting Common Issues

### 1. EasyOCR Model Download / PyTorch Warnings
* **Issue:** EasyOCR takes time on first run or hangs.
* **Fix:** EasyOCR downloads detection/recognition neural network weights (`~100MB`) to `~/.EasyOCR/` on the initial execution. Ensure internet connectivity on the first run.
* **CPU vs GPU:** If you do not have CUDA installed, EasyOCR automatically runs on CPU without any extra setup.

### 2. OpenCV Headless vs GUI Conflict
* **Issue:** `ImportError: cannot import name '...' from 'cv2'` or Qt platform plugin errors.
* **Fix:** Use `opencv-python-headless` instead of `opencv-python` to avoid desktop GUI dependencies in server/Streamlit environments:
  ```bash
  pip uninstall opencv-python opencv-python-headless -y
  pip install opencv-python-headless
  ```

### 3. Windows Long Path Errors (PyTorch / EasyOCR)
* **Issue:** `[Errno 2] No such file or directory` during model extraction on Windows.
* **Fix:** Enable long paths in Windows registry or run terminal as Administrator:
  ```cmd
  reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
  ```

### 4. Groq Model Not Found (404)
* **Issue:** Specific model deprecated or unavailable on your Groq tier.
* **Fix:** `llm_parser.py` includes automatic fallback across active models (`openai/gpt-oss-120b`, `qwen/qwen3.6-27b`, `llama-3.3-70b-versatile`).
