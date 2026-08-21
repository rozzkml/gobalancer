"""
LLM Gateway v2 - Load Balancer Multi Provider
Support: Groq, Google Gemini, DeepSeek, Ollama
- Dynamic API key detection dari environment variables
- Compatible dengan OpenAI API format
- Ready untuk Vercel deployment
"""

import os
import time
import httpx
from typing import Optional
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="GoBalancer", version="2.0.0")

# CORS supaya bisa dipanggil dari mana saja (OpenClaw, Hermes, dll)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# AUTO-DETECT API KEYS DARI ENVIRONMENT
# Tambah GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3, dst
# di Vercel Environment Variables — otomatis terdeteksi semua
# ============================================================

def load_keys(prefix: str) -> list[str]:
    """
    Auto-detect semua key dengan prefix tertentu.
    Contoh prefix 'GROQ_API_KEY' akan detect:
    GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3, ... dst
    Juga detect GROQ_API_KEY (tanpa angka) sebagai fallback.
    """
    keys = []

    # Cek tanpa nomor dulu (GROQ_API_KEY)
    single = os.getenv(prefix, "").strip()
    if single:
        keys.append(single)

    # Cek dengan nomor 1-50
    for i in range(1, 51):
        key = os.getenv(f"{prefix}_{i}", "").strip()
        if key:
            keys.append(key)

    # Hapus duplikat, pertahankan urutan
    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    return unique_keys


# ============================================================
# KONFIGURASI PROVIDER
# ============================================================

def build_providers() -> dict:
    return {
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_keys": load_keys("GEMINI_API_KEY"),
            "models": {
                # ── High RPD only (free-tier). Gemma = highest RPD 14.4K/day ──
                "gemma-4-31b-it": "gemma-4-31b-it",
                "gemma-4-26b-a4b-it": "gemma-4-26b-a4b-it",
                "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
            },
            "default_model": "gemma-4-31b-it",
            "rpm_limit": 15,
        },
    }

PROVIDERS = build_providers()

# ============================================================
# GATEWAY AUTH (opsional tapi direkomendasikan)
# Set GATEWAY_KEYS di env, comma-separated, cth:
#   GATEWAY_KEYS=sk-abc123,sk-def456
# Kalau kosong → gateway OPEN (siapa aja bisa pakai key provider lo).
# Client wajib kirim header: Authorization: Bearer <gateway_key>
# ============================================================

def load_gateway_keys() -> list[str]:
    raw = os.getenv("GATEWAY_KEYS", "").strip()
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    # Fallback: dukung juga GATEWAY_KEY_1..50 (konsisten sama pola provider)
    single = os.getenv("GATEWAY_KEY", "").strip()
    if single:
        keys.append(single)
    for i in range(1, 51):
        k = os.getenv(f"GATEWAY_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    # dedup, pertahankan urutan
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

GATEWAY_KEYS = load_gateway_keys()

def verify_gateway_key(authorization: Optional[str] = Header(None)):
    # Gateway terbuka kalau belum ada GATEWAY_KEYS di-set
    if not GATEWAY_KEYS:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            401,
            detail={"error": "Missing bearer token", "hint": "Kirim header: Authorization: Bearer <gateway_key>"},
        )
    token = authorization.split(" ", 1)[1].strip()
    if token not in GATEWAY_KEYS:
        raise HTTPException(401, detail={"error": "Invalid gateway key"})

# ============================================================
# MODEL ALIASES
# ============================================================

MODEL_ALIASES = {
    # Gemini (shortcut)
    "gemini":            "gemini/gemini-3.5-flash-lite",  # default → high RPD
    "gemini-lite":       "gemini/gemini-3.5-flash-lite",
    "gemini-flash-lite": "gemini/gemini-3.5-flash-lite",
    "gemma":             "gemini/gemma-4-31b-it",
}

# ============================================================
# RATE LIMIT TRACKER (in-memory)
# ============================================================

class RateLimitTracker:
    def __init__(self):
        self.requests: dict[str, list[float]] = defaultdict(list)

    def _clean(self, key: str):
        now = time.time()
        self.requests[key] = [t for t in self.requests[key] if now - t < 60]

    def can_request(self, key: str, rpm_limit: int) -> bool:
        self._clean(key)
        return len(self.requests[key]) < rpm_limit

    def record(self, key: str):
        self.requests[key].append(time.time())

    def wait_time(self, key: str, rpm_limit: int) -> float:
        self._clean(key)
        window = self.requests[key]
        if len(window) < rpm_limit:
            return 0.0
        return max(0.0, 60 - (time.time() - min(window)))

tracker = RateLimitTracker()

# ============================================================
# KEY ROTATOR (round-robin)
# ============================================================

class KeyRotator:
    def __init__(self):
        self.indices: dict[str, int] = defaultdict(int)

    def get_key(self, provider_name: str, rpm_limit: int) -> Optional[str]:
        keys = PROVIDERS[provider_name]["api_keys"]
        if not keys:
            return None

        start = self.indices[provider_name]
        for i in range(len(keys)):
            idx = (start + i) % len(keys)
            key = keys[idx]
            tracker_key = f"{provider_name}:{key}"
            if tracker.can_request(tracker_key, rpm_limit):
                self.indices[provider_name] = (idx + 1) % len(keys)
                tracker.record(tracker_key)
                return key
        return None

rotator = KeyRotator()

# ============================================================
# STATS
# ============================================================

stats: dict = defaultdict(lambda: {"success": 0, "error": 0, "total_tokens": 0})

# ============================================================
# MODELS
# ============================================================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "gemini"
    messages: list[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

# ============================================================
# HELPER: RESOLVE MODEL
# ============================================================

def resolve_model(model_str: str) -> tuple[str, str, str]:
    # Resolve alias
    if model_str in MODEL_ALIASES:
        model_str = MODEL_ALIASES[model_str]

    # Format "provider/model"
    if "/" in model_str:
        provider_name, model_alias = model_str.split("/", 1)
        if provider_name not in PROVIDERS:
            raise HTTPException(400, f"Provider '{provider_name}' tidak dikenal. Tersedia: {list(PROVIDERS.keys())}")
        provider = PROVIDERS[provider_name]
        api_model = provider["models"].get(model_alias, model_alias)
        return provider_name, api_model, provider["base_url"]

    # Cari di semua provider
    for pname, pconfig in PROVIDERS.items():
        if model_str in pconfig["models"]:
            return pname, pconfig["models"][model_str], pconfig["base_url"]

    # Default ke Gemini (satu-satunya provider aktif)
    return "gemini", model_str, PROVIDERS["gemini"]["base_url"]

# ============================================================
# ENDPOINT: CHAT COMPLETIONS
# ============================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, _auth: None = Depends(verify_gateway_key)):
    provider_name, api_model, base_url = resolve_model(request.model)
    provider = PROVIDERS[provider_name]
    rpm_limit = provider["rpm_limit"]

    api_key = rotator.get_key(provider_name, rpm_limit)
    if not api_key:
        keys = provider["api_keys"]
        if not keys:
            raise HTTPException(
                503,
                detail={
                    "error": f"Tidak ada API key untuk '{provider_name}'",
                    "hint": f"Tambahkan {provider_name.upper()}_API_KEY_1 di environment variables"
                }
            )
        wait = min(
            tracker.wait_time(f"{provider_name}:{k}", rpm_limit)
            for k in keys
        )
        raise HTTPException(
            429,
            detail={
                "error": "Semua key sedang rate limited",
                "provider": provider_name,
                "retry_after_seconds": round(wait, 1)
            }
        )

    payload = {
        "model": api_model,
        "messages": [m.dict() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            if request.stream:
                async def stream_gen():
                    async with client.stream(
                        "POST",
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                stats[provider_name]["success"] += 1
                return StreamingResponse(stream_gen(), media_type="text/event-stream")

            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )

            if resp.status_code == 429:
                stats[provider_name]["error"] += 1
                raise HTTPException(429, detail={"error": "Rate limited oleh provider", "provider": provider_name})

            if resp.status_code != 200:
                stats[provider_name]["error"] += 1
                raise HTTPException(resp.status_code, detail=resp.text)

            data = resp.json()
            stats[provider_name]["success"] += 1

            if "usage" in data:
                stats[provider_name]["total_tokens"] += data["usage"].get("total_tokens", 0)

            # Inject info gateway ke response
            data["_gateway"] = {
                "provider": provider_name,
                "model_requested": request.model,
                "model_used": api_model,
            }
            return JSONResponse(data)

        except httpx.TimeoutException:
            stats[provider_name]["error"] += 1
            raise HTTPException(504, detail=f"Timeout dari '{provider_name}'")
        except httpx.RequestError as e:
            stats[provider_name]["error"] += 1
            raise HTTPException(502, detail=f"Koneksi gagal: {str(e)}")

# ============================================================
# ENDPOINT: LIST MODELS
# ============================================================

@app.get("/v1/models")
async def list_models():
    """
    Hanya tampilkan model yang benar-benar bisa dipakai:
    provider harus punya minimal 1 API key ke-load.
    """
    models = []
    usable_aliases = set()
    for pname, pconfig in PROVIDERS.items():
        key_count = len(pconfig["api_keys"])
        if key_count == 0:
            continue  # skip provider tanpa key -> gak bisa dipakai
        for alias, api_name in pconfig["models"].items():
            models.append({
                "id": f"{pname}/{alias}",
                "object": "model",
                "provider": pname,
                "api_model": api_name,
            })
            usable_aliases.add(f"{pname}/{alias}")
    # Alias cuma ditampilin kalau target-nya usable
    for alias, target in MODEL_ALIASES.items():
        if target in usable_aliases:
            models.append({
                "id": alias,
                "object": "model",
                "alias_for": target,
            })
    return {"object": "list", "data": models}

# ============================================================
# ENDPOINT: STATS
# ============================================================

@app.get("/stats")
async def get_stats():
    result = {}
    for pname, pconfig in PROVIDERS.items():
        keys = pconfig["api_keys"]
        key_status = []
        for idx, key in enumerate(keys, 1):
            kid = f"{pname}:{key}"
            reqs = len([t for t in tracker.requests.get(kid, []) if time.time() - t < 60])
            key_status.append({
                # No key fragment exposed — anonymous slot label only
                "slot": f"key #{idx}",
                "requests_last_minute": reqs,
                "rpm_limit": pconfig["rpm_limit"],
                "available": tracker.can_request(kid, pconfig["rpm_limit"]),
            })
        result[pname] = {
            "keys_loaded": len(keys),
            "success": stats[pname]["success"],
            "error": stats[pname]["error"],
            "total_tokens": stats[pname]["total_tokens"],
            "keys": key_status,
        }
    return result

# ============================================================
# ENDPOINT: HEALTH & ROOT
# ============================================================

@app.get("/health")
async def health():
    summary = {
        pname: f"{len(pconfig['api_keys'])} keys loaded"
        for pname, pconfig in PROVIDERS.items()
    }
    return {"status": "ok", "providers": summary}

@app.get("/")
async def root():
    from fastapi.responses import HTMLResponse
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "index.html"),
        os.path.join(here, "static", "index.html"),
        os.path.join(os.getcwd(), "index.html"),
        "/var/task/index.html",
        os.path.join(os.path.dirname(here), "index.html"),
    ]
    for candidate in candidates:
        try:
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    return HTMLResponse(f.read())
        except Exception:
            continue
    return {
        "name": "GoBalancer",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "stats": "GET /stats",
            "health": "GET /health",
        },
        "aliases": MODEL_ALIASES,
    }

# Mount static files — taruh di bawah semua route
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
