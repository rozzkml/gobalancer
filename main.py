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
from typing import Optional, Any
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from normalize import strip_thoughts, ThoughtStripper


def _sse_content_delta(tmpl, text):
    """Build one SSE `data:` line carrying a content delta, cloned from a
    real upstream chunk `tmpl` so id/model/created stay consistent."""
    import json as _json
    if tmpl:
        obj = {
            "id": tmpl.get("id", ""),
            "object": "chat.completion.chunk",
            "created": tmpl.get("created", 0),
            "model": tmpl.get("model", ""),
            "choices": [{"index": 0, "finish_reason": None,
                         "delta": {"role": "assistant", "content": text}}],
        }
    else:
        obj = {"object": "chat.completion.chunk",
               "choices": [{"index": 0, "finish_reason": None,
                            "delta": {"role": "assistant", "content": text}}]}
    return ("data: " + _json.dumps(obj) + "\n\n").encode()

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

# Track how many raw env vars were set vs unique keys, per prefix.
KEY_LOAD_STATS: dict[str, dict] = {}

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
    dup_fingerprints = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)
        else:
            dup_fingerprints.append(f"{k[:4]}…{k[-4:]}" if len(k) >= 10 else "…")

    KEY_LOAD_STATS[prefix] = {
        "raw_set": len(keys),
        "unique_loaded": len(unique_keys),
        "duplicates_removed": len(keys) - len(unique_keys),
        "duplicate_fingerprints": dup_fingerprints,
    }

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
                # ── Agent-capable only. Gemma dropped: 16K input-TPM cap can't
                #    fit an agent-sized system prompt (429 on every turn). ──
                "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
            },
            "default_model": "gemini-3.5-flash-lite",
            "rpm_limit": 15,
        },
        "opencode-zen": {
            "base_url": "https://opencode.ai/zen/v1",
            "api_keys": load_keys("OPENCODE_ZEN_API_KEY"),
            # Models are discovered dynamically (free-tier only) — see zen_sync_models().
            "models": {},
            "default_model": "nemotron-3-ultra-free",
            "rpm_limit": 15,
            "dynamic_free": True,
        },
    }

PROVIDERS = build_providers()

# ============================================================
# OPENCODE-ZEN: DYNAMIC FREE-MODEL SYNC
# The zen free-model list changes over time (models get added/removed,
# e.g. deepseek-v4-flash-free went unavailable). Instead of hardcoding,
# we fetch GET /models, keep only ids ending in "-free", probe each once
# with a tiny request, and cache the working set. Refreshed on a TTL.
# ============================================================

ZEN_CACHE: dict = {"models": {}, "ts": 0.0, "listed": [], "dead": []}
ZEN_TTL = 900  # re-list at most every 15 min per warm container
# Runtime death registry: ids that returned "unavailable"/error at request time.
# Self-heals — an id is re-tried once its TTL entry expires.
ZEN_DEAD: dict = {}      # {model_id: expiry_ts}
ZEN_DEAD_TTL = 1800      # keep a dead model out for 30 min, then re-test lazily
# Seed with ids we already confirmed broken during setup.
_ZEN_SEED_DEAD = {"deepseek-v4-flash-free", "muse-spark-1.2-contributor-free"}

def _zen_is_dead(mid: str) -> bool:
    exp = ZEN_DEAD.get(mid)
    if exp is None:
        return False
    if time.time() >= exp:      # expired → give it another chance
        ZEN_DEAD.pop(mid, None)
        return False
    return True

def zen_mark_dead(mid: str):
    ZEN_DEAD[mid] = time.time() + ZEN_DEAD_TTL
    ZEN_CACHE["models"].pop(mid, None)

def zen_sync_models(force: bool = False) -> dict:
    """Cheap sync: GET /models (1 call), keep only live '-free' ids. TTL-gated.
    Broken ids are pruned lazily at request time via ZEN_DEAD (self-healing)."""
    prov = PROVIDERS.get("opencode-zen")
    if not prov or not prov.get("dynamic_free"):
        return {}
    keys = prov["api_keys"]
    if not keys:
        return {}
    now = time.time()
    if not force and (now - ZEN_CACHE["ts"] < ZEN_TTL) and ZEN_CACHE["models"]:
        return ZEN_CACHE["models"]
    try:
        with httpx.Client(timeout=20) as c:
            r = c.get(f"{prov['base_url']}/models",
                      headers={"Authorization": f"Bearer {keys[0]}"})
        listed = [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return ZEN_CACHE["models"]  # keep last good cache on failure

    free_ids = [m for m in listed if m.endswith("-free")]
    working = {m: m for m in free_ids if not _zen_is_dead(m)}
    prov["models"] = working
    if prov.get("default_model") not in working and working:
        prov["default_model"] = next(iter(working))
    ZEN_CACHE.update({"models": working, "ts": now, "listed": free_ids,
                      "dead": sorted(ZEN_DEAD.keys())})
    return working

# Seed dead set (no network at import; keeps cold-start instant).
for _m in _ZEN_SEED_DEAD:
    ZEN_DEAD[_m] = time.time() + ZEN_DEAD_TTL

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
    "gemini":            "gemini/gemini-3.5-flash-lite",  # bare default → agent-capable
}

# Per-model RPM limit per key (Google free-tier: Gemma 30/min, flash-lite 15/min).
# Buckets are tracked per (provider, model, key) so each model uses its own quota.
MODEL_RPM = {
    "gemini-3.5-flash-lite": 15,
    "gemini-3.1-flash-lite": 15,
}
DEFAULT_RPM = 15

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

    def get_key(self, provider_name: str, api_model: str, rpm_limit: int) -> Optional[str]:
        keys = PROVIDERS[provider_name]["api_keys"]
        if not keys:
            return None

        # Round-robin per (provider, model) so each model rotates independently.
        rr_key = f"{provider_name}:{api_model}"
        start = self.indices[rr_key]
        for i in range(len(keys)):
            idx = (start + i) % len(keys)
            key = keys[idx]
            # Per-model bucket: each model consumes its own quota per key.
            tracker_key = f"{provider_name}:{api_model}:{key}"
            if tracker.can_request(tracker_key, rpm_limit):
                self.indices[rr_key] = (idx + 1) % len(keys)
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
    # content optional: assistant tool-call turns and some tool messages carry no text
    content: Optional[Any] = None
    # OpenAI tool-calling passthrough fields
    name: Optional[str] = None
    tool_calls: Optional[list[Any]] = None
    tool_call_id: Optional[str] = None

class ChatRequest(BaseModel):
    model: str = "gemini"
    messages: list[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    # tool-calling passthrough
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None

# ============================================================
# HELPER: RESOLVE MODEL
# ============================================================

def resolve_model(model_str: str) -> tuple[str, str, str]:
    # Resolve alias
    if model_str in MODEL_ALIASES:
        model_str = MODEL_ALIASES[model_str]

    # Keep the opencode-zen free-model list fresh (TTL-gated, ~1 call/15min).
    if "opencode-zen" in PROVIDERS and PROVIDERS["opencode-zen"].get("dynamic_free"):
        try:
            zen_sync_models()
        except Exception:
            pass

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
    # Per-model RPM: Gemma 30/min/key, flash-lite 15/min/key (Google free-tier).
    rpm_limit: int = MODEL_RPM.get(api_model, provider.get("rpm_limit") or DEFAULT_RPM)

    api_key = rotator.get_key(provider_name, api_model, rpm_limit)
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
            tracker.wait_time(f"{provider_name}:{api_model}:{k}", rpm_limit)
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
        "messages": [m.dict(exclude_none=True) for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }
    # Forward tool-calling fields when the client sends them (agentic clients like Hermes).
    if request.tools is not None:
        payload["tools"] = request.tools
    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if request.stream:
        # NOTE: the AsyncClient MUST live inside the generator. If we open it in
        # an `async with` here, it closes the moment this handler returns the
        # StreamingResponse — before the ASGI server starts iterating the
        # generator — yielding a 200 with an empty body (the classic Vercel
        # "streams nothing" bug). Owning the client inside gen() keeps it alive
        # for the whole stream. Real incremental streaming on Vercel also
        # requires Fluid Compute enabled + the anti-buffering headers below.
        async def stream_gen():
            import json as _json
            stripper = ThoughtStripper()
            reasoning_buf = []          # accumulate reasoning in case content stays empty
            emitted_content = False     # did we forward ANY real content delta?
            tmpl = None                 # a sample chunk dict to synthesize a fallback delta
            try:
                async with httpx.AsyncClient(timeout=None) as sclient:
                    async with sclient.stream(
                        "POST",
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status_code != 200:
                            stats[provider_name]["error"] += 1
                            body = await resp.aread()
                            yield body
                            return
                        stats[provider_name]["success"] += 1
                        async for raw in resp.aiter_lines():
                            if not raw:
                                yield b"\n"
                                continue
                            if not raw.startswith("data:"):
                                # SSE comments / keep-alives — pass through
                                yield (raw + "\n").encode()
                                continue
                            data_str = raw[5:].lstrip()
                            if data_str == "[DONE]":
                                # Before closing: if the model never produced
                                # visible content but streamed reasoning, promote
                                # the reasoning so agent clients aren't left blank.
                                tail = stripper.flush()
                                if tail:
                                    emitted_content = True
                                    yield _sse_content_delta(tmpl, tail)
                                if not emitted_content and reasoning_buf and tmpl:
                                    yield _sse_content_delta(tmpl, "".join(reasoning_buf))
                                yield b"data: [DONE]\n\n"
                                continue
                            try:
                                obj = _json.loads(data_str)
                            except Exception:
                                yield (raw + "\n\n").encode()
                                continue
                            ch = (obj.get("choices") or [{}])
                            if not ch:
                                yield (raw + "\n\n").encode()
                                continue
                            delta = ch[0].get("delta") or {}
                            if tmpl is None:
                                tmpl = obj
                            # capture reasoning (both spellings) for fallback
                            rc = delta.get("reasoning_content") or delta.get("reasoning")
                            if rc:
                                reasoning_buf.append(rc)
                            # strip <thought> blocks from visible content
                            if delta.get("content"):
                                clean = stripper.feed(delta["content"])
                                if clean:
                                    emitted_content = True
                                    delta["content"] = clean
                                    yield ("data: " + _json.dumps(obj) + "\n\n").encode()
                                # if clean is empty (fully inside a thought), drop
                                # this chunk entirely — don't forward empty noise
                                continue
                            # non-content chunk (role/finish/usage/tool_calls) —
                            # forward untouched
                            yield ("data: " + _json.dumps(obj) + "\n\n").encode()
            except Exception as e:
                stats[provider_name]["error"] += 1
                yield (b"data: " + _json.dumps({
                    "error": {"message": f"gateway stream error: {e}"}
                }).encode() + b"\n\n")
                yield b"data: [DONE]\n\n"
        return StreamingResponse(
            stream_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # disable proxy buffering
            },
        )

    async with httpx.AsyncClient(timeout=60) as client:
        try:
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
                # opencode-zen free models can vanish ("Model is unavailable").
                # Mark dead so the dynamic sync drops it until its TTL expires.
                if provider_name == "opencode-zen" and ("unavailable" in resp.text.lower()
                                                         or resp.status_code in (400, 404)):
                    zen_mark_dead(api_model)
                raise HTTPException(resp.status_code, detail=resp.text)

            data = resp.json()
            stats[provider_name]["success"] += 1

            # ── Normalize output so EVERY model is agent-clean ──
            # 1) strip <thought>/<think> blocks that leak into content
            # 2) if content is empty but reasoning_content exists (reasoning-only
            #    models like hy3), promote reasoning into content so agent
            #    clients (Hermes/OpenClaw read `content`) aren't left blank.
            # tool_calls turns are left untouched (empty content is valid there).
            try:
                for _c in data.get("choices", []):
                    _m = _c.get("message")
                    if not isinstance(_m, dict):
                        continue
                    if _m.get("tool_calls"):
                        continue  # valid empty-content tool turn
                    _txt = _m.get("content")
                    if isinstance(_txt, str) and _txt:
                        _m["content"] = strip_thoughts(_txt)
                    # promote reasoning if content ended up empty
                    if not (_m.get("content") or "").strip():
                        _r = _m.get("reasoning_content") or _m.get("reasoning")
                        if _r:
                            _m["content"] = strip_thoughts(_r) if isinstance(_r, str) else _r
            except Exception:
                pass

            # Zen sometimes 200s with an empty {role} message (dead-ish model).
            # But a valid tool-call turn also has empty content (tool_calls only) —
            # don't mark those dead.
            if provider_name == "opencode-zen":
                _ch = (data.get("choices") or [{}])[0]
                _msg = _ch.get("message", {}) or {}
                if not (_msg.get("content") or _msg.get("reasoning_content") or _msg.get("tool_calls")):
                    zen_mark_dead(api_model)

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
    # Refresh opencode-zen free-model list (TTL-gated).
    try:
        zen_sync_models()
    except Exception:
        pass
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
            # Per-model buckets: aggregate usage across all models on this key.
            now = time.time()
            per_model = {}
            total_reqs = 0
            for m, mrpm in MODEL_RPM.items():
                kid = f"{pname}:{m}:{key}"
                r = len([t for t in tracker.requests.get(kid, []) if now - t < 60])
                total_reqs += r
                per_model[m] = {"rpm": r, "rpm_limit": mrpm, "available": r < mrpm}
            key_status.append({
                # No key fragment exposed — anonymous slot label only
                "slot": f"key #{idx}",
                # Masked fingerprint: first4…last4 only — safe to show, lets you
                # spot duplicate keys without exposing the full secret.
                "fingerprint": (f"{key[:4]}…{key[-4:]}" if len(key) >= 10 else "…"),
                "requests_last_minute": total_reqs,
                "per_model": per_model,
            })
        result[pname] = {
            "keys_loaded": len(keys),
            "key_load": KEY_LOAD_STATS.get(f"{pname.upper()}_API_KEY", {}),
            "success": stats[pname]["success"],
            "error": stats[pname]["error"],
            "total_tokens": stats[pname]["total_tokens"],
            "keys": key_status,
        }
        # opencode-zen: surface dynamic free-model sync state.
        if pconfig.get("dynamic_free"):
            result[pname]["dynamic_free"] = {
                "models_live": sorted(pconfig["models"].keys()),
                "listed_free": ZEN_CACHE.get("listed", []),
                "dead": sorted(ZEN_DEAD.keys()),
                "default_model": pconfig.get("default_model"),
                "synced_ago_s": round(time.time() - ZEN_CACHE["ts"], 1) if ZEN_CACHE["ts"] else None,
                "ttl_s": ZEN_TTL,
            }
    return result

@app.post("/zen/sync")
async def zen_sync():
    """Force-refresh the opencode-zen free-model list (bypasses TTL)."""
    if "opencode-zen" not in PROVIDERS:
        raise HTTPException(404, "opencode-zen provider not configured")
    working = zen_sync_models(force=True)
    return {
        "synced": True,
        "models_live": sorted(working.keys()),
        "listed_free": ZEN_CACHE.get("listed", []),
        "dead": sorted(ZEN_DEAD.keys()),
        "default_model": PROVIDERS["opencode-zen"].get("default_model"),
    }

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
