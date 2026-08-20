"""
Entry point untuk Vercel Serverless.
Vercel hanya menganggap file di dalam folder /api/ sebagai serverless function.
File ini mengimpor `app` dari main.py yang ada di root repo.
"""
import os
import sys

# Pastikan root repo (parent dari folder /api) ada di sys.path
# supaya `from main import app` selalu ketemu di runtime Vercel.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import app  # noqa: E402

# Vercel Python runtime (ASGI) akan otomatis mendeteksi variabel `app`.
