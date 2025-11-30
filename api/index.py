import os
import json
import re
import base64
import uuid
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
import httpx
import fitz
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY required")

sessions: Dict[str, dict] = {}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def call_llm(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 4096
            }
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

async def analyze_image(b64: str) -> str:
    if not OPENROUTER_API_KEY:
        return "Image analysis unavailable."
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model": "anthropic/claude-3.5-sonnet",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Describe this figure clearly."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]}]
                }
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except:
        return "Image analysis failed."

def extract_content(file_bytes: bytes, filename: str):
    text = ""
    images = []
    if filename.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n\n"
                for img in page.get_images(full=True):
                    xref = img[0]
                    base = doc.extract_image(xref)
                    if base["ext"] in ["png", "jpeg", "jpg"]:
                        images.append(base["image"])
            doc.close()
        except Exception as e:
            text = f"PDF error: {e}"
    return text.strip(), images

async def minimal_edit(text: str):
    prompt = f'''Fix only real spelling/grammar errors. Never rephrase.
Return ONLY valid JSON:

{{
  "corrected": "full corrected text",
  "corrections": [
    {{"original": "recieve", "corrected": "receive"}}
  ]
}}

Text:
{text[:14000]}'''

    try:
        raw = await call_llm(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return text, []
        data = json.loads(match.group(0))
        corrected = data.get("corrected", text)
        corrections = data.get("corrections", [])
        for c in corrections:
            pattern = re.compile(re.escape(c["original"]), re.I)
            corrected = pattern.sub(
                lambda m: f'<span style="color:red; text-decoration:underline wavy red; font-weight:bold;">{m.group(0)}</span>',
                corrected
            )
        return corrected, corrections
    except:
        return text, []

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text, images = extract_content(content, file.filename)
    edited_text, corrections = await minimal_edit(text)
    
    seen = set()
    unique_corrections = []
    for c in corrections:
        key = (c["original"].lower(), c["corrected"].lower())
        if key not in seen:
            seen.add(key)
            unique_corrections.append(c)

    # Image analysis
    figures = []
    if images and OPENROUTER_API_KEY:
        for i, img in enumerate(images[:6]):
            b64 = base64.b64encode(img).decode()
            analysis = await analyze_image(b64)
            figures.append({
                "id": i + 1,
                "image": f"data:image/png;base64,{b64}",
                "analysis": analysis
            })

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"text": text}

    return JSONResponse({
        "session_id": session_id,
        "original_text": text[:15000] + ("..." if len(text) > 15000 else ""),
        "corrected_text": edited_text,
        "corrections": unique_corrections,
        "figures": figures
    })

@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in sessions:
        await websocket.close(code=1008)
        return
    paper = sessions[session_id]["text"][:20000]
    try:
        while True:
            q = await websocket.receive_text()
            prompt = f"Paper:\n{paper}\n\nQuestion: {q}\nAnswer:"
            ans = await call_llm(prompt)
            await websocket.send_text(ans)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.send_text("Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("script:app", host="0.0.0.0", port=8000, reload=True)