"""
Interactive PDF Chatbot (Flask app using Gemini API)
Production-ready version for Google Cloud Run
"""
import os
import io
import time
import logging
from typing import List, Tuple

from flask import Flask, request, jsonify, render_template_string, session
from flask_session import Session  # Import Flask-Session
from PyPDF2 import PdfReader
import nltk

# Optional extras for PDF text extraction (OCR)
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

# NLTK
nltk.download('punkt', quiet=True)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("pdf_chat_app")

# --- Configuration ---

# 1. Session Configuration
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key-dev-only')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session'
app.config['SESSION_PERMANENT'] = False
Session(app)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# 2. App Configuration
PORT = int(os.environ.get('PORT', 8080))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ALLOW_OFFLINE = os.environ.get('ALLOW_OFFLINE', '0') == '1'

# --- MODEL UPDATE HERE ---
# Prioritizing gemini-2.5-pro as requested
MODEL_NAME = os.environ.get('GEMINI_MODEL', 'gemini-2.5-pro')
# --- END OF UPDATE ---

MAX_UPLOAD_MB = int(os.environ.get('MAX_UPLOAD_MB', 25))

# 3. Configure genai
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        log.info(f"Configured google.generativeai (Primary model: {MODEL_NAME})")
    except Exception as e:
        log.warning("Failed to configure google.generativeai: %s", e)
        genai = None
else:
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set. App will fail to answer questions.")
    if genai is None:
        log.info("google-generativeai not installed or failed to import.")


# ----- PDF Extraction Utilities (Unchanged) -----

def safe_read_file_storage(f) -> bytes:
    """Safely read bytes from a Flask FileStorage object."""
    data = f.read()
    try:
        f.seek(0)
    except Exception:
        pass
    return data

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, List[bytes]]:
    """Extracts text using PyPDF2 and optionally images for OCR."""
    page_images = []
    text_parts = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        log.warning("PyPDF2 failed to read PDF: %s", e)
        return "", []

    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
            if txt.strip():
                text_parts.append(txt)
        except Exception:
            continue

    if not text_parts and convert_from_bytes is not None:
        log.info("No text extracted by PyPDF2. Attempting OCR...")
        try:
            imgs = convert_from_bytes(file_bytes, dpi=200)
            for im in imgs:
                bio = io.BytesIO()
                im.save(bio, format='PNG')
                page_images.append(bio.getvalue())
        except Exception as e:
            log.warning("pdf2image conversion failed: %s", e)

    return "\n".join(text_parts), page_images

def ocr_pdf_images(page_images: List[bytes]) -> str:
    """Performs OCR on a list of page images."""
    if not page_images or Image is None or pytesseract is None:
        return ''
    log.info(f"Performing OCR on {len(page_images)} images...")
    texts = []
    for i, b in enumerate(page_images):
        try:
            img = Image.open(io.BytesIO(b))
            txt = pytesseract.image_to_string(img)
            if txt:
                texts.append(txt)
        except Exception as e:
            log.warning("OCR failed for page image %d: %s", i, e)
    log.info("OCR complete.")
    return "\n".join(texts)

# ----- Model Generation Utilities -----

# --- MODEL UPDATE HERE ---
# Updated fallback list
FALLBACK_MODELS = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
# --- END OF UPDATE ---

def gemini_generate(prompt: str, max_output_tokens: int = 1024) -> str:
    """Resilient wrapper: tries configured model(s); falls back to offline stub if allowed."""
    if genai is None:
        if ALLOW_OFFLINE:
            return f"[OFFLINE-STUB] This is an offline answer based on the document."
        log.error("Generative model not configured and offline mode not enabled.")
        return "Error: The generative model is not configured."

    # This line dynamically creates the list, starting with MODEL_NAME
    # It will try 'gemini-2.5-pro' first, then the fallbacks.
    candidates = ([MODEL_NAME] if MODEL_NAME else []) + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
    tried = []
    for m in candidates:
        if not m:
            continue
        tried.append(m)
        try:
            model = genai.GenerativeModel(m)
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.1
            )
            log.info(f"Attempting to generate content with model: {m}")
            resp = model.generate_content(prompt, generation_config=generation_config)
            text = getattr(resp, 'text', None)
            if text:
                log.info(f"Successfully generated content with {m}")
                return text
            log.warning("Model %s returned empty response or was blocked. Response: %s", m, resp)
            return f"Error: Model {m} returned an empty or blocked response."
        except Exception as e:
            log.warning("Model %s failed: %s", m, e)
            continue

    if ALLOW_OFFLINE:
        return f"[OFFLINE-FALLBACK] All models failed (tried {', '.join(tried)})."
    
    log.error("All model attempts failed (tried: %s)", ", ".join(tried))
    return f"Error: All generative models failed. (Tried: {', '.join(tried)})"


def answer_question_from_context(context: str, question: str) -> str:
    """
    Creates a prompt and calls Gemini to answer a question based on context.
    """
    prompt = (
        "You are a helpful assistant. You will be given a document's text as context. "
        "Your task is to answer the user's question based *only* on the provided context.\n"
        "Do not use any external knowledge. Do not make up information.\n"
        "If the answer is not found in the context, state that clearly "
        "(e.g., 'The provided document does not contain information on this topic.').\n"
        "\n--- DOCUMENT CONTEXT ---\n"
        f"{context}"
        "\n--- END OF CONTEXT ---\n"
        "\nUSER QUESTION:\n"
        f"{question}"
    )
    return gemini_generate(prompt, max_output_tokens=1024)


# ----- Flask Routes (Unchanged) -----

@app.route('/')
def index():
    """Serves a simple check to see if the server is alive."""
    return "Flask PDF Chat API is running."

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model': MODEL_NAME if genai else ('offline' if ALLOW_OFFLINE else 'not-configured'),
        'pdf2image': bool(convert_from_bytes),
        'pytesseract': bool(pytesseract),
        'session_type': app.config.get('SESSION_TYPE'),
    })

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Handles PDF upload. Extracts text and stores it in the session.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    start_time = time.time()
    try:
        content = safe_read_file_storage(f)
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            return jsonify({'error': f'file too large (>{MAX_UPLOAD_MB} MB)'}), 400

        extracted_text, page_images = extract_text_from_pdf(content)
        ocr_text = ''
        if (not extracted_text.strip()) and page_images and pytesseract is not None:
            ocr_text = ocr_pdf_images(page_images)

        full_text = (extracted_text + "\n" + ocr_text).strip()
        
        if not full_text:
            return jsonify({'error': 'no text extracted from PDF (and OCR not available or failed).'}), 400

        session['pdf_text'] = full_text
        session['filename'] = f.filename
        
        log.info(f"Processed and stored '{f.filename}' in session. Text length: {len(full_text)} chars.")
        
        return jsonify({
            'status': 'success',
            'message': f'File "{f.filename}" processed.',
            'filename': f.filename,
            'text_length': len(full_text),
            'elapsed_sec': round(time.time() - start_time, 2)
        })

    except Exception as e:
        log.exception("PDF processing failed: %s", e)
        return jsonify({'error': f'An error occurred during processing: {e}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Answers a user's question based on the text stored in the session.
    """
    pdf_text = session.get('pdf_text')
    filename = session.get('filename', 'the document')

    if not pdf_text:
        return jsonify({'error': 'No PDF uploaded. Please upload a PDF first.'}), 400

    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'no question provided'}), 400

    log.info(f"Answering question about '{filename}': {question[:50]}...")
    
    try:
        answer = answer_question_from_context(pdf_text, question)
        return jsonify({
            'answer': answer,
            'question': question,
            'context_from': filename
        })
    except Exception as e:
        log.exception("Answer generation failed: %s", e)
        return jsonify({'error': f'An error occurred while generating the answer: {e}'}), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    """
    Clears all data from the user's session.
    """
    try:
        filename = session.get('filename', 'any')
        session.clear()
        log.info(f"Session cleared. Removed data for '{filename}'.")
        return jsonify({
            'status': 'cleared',
            'message': 'Session cleared. All PDF data has been erased.'
        })
    except Exception as e:
        log.exception("Session clear failed: %s", e)
        return jsonify({'error': f'Could not clear session: {e}'}), 500

# ----- Run the App -----
if __name__ == '__main__':
    # This block is for LOCAL TESTING ONLY.
    # Gunicorn will be used in production.
    log.info(f"Starting Flask dev server on http://0.0.0.0:{PORT}")
    app.run(host='0.0.0.0', port=PORT, threaded=True, debug=True)

