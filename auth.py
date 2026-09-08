import os
from dotenv import load_dotenv
from supabase import create_client
from fastapi import Header, HTTPException

load_dotenv()

_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in .env — "
                "Supabase client cannot be initialized."
            )
        _supabase_client = create_client(url, key)
    return _supabase_client


def verify_api_key(x_api_key: str = Header(...)):
    """FastAPI dependency — runs before the endpoint body executes.
    Rejects the request early if the key is missing, unknown, or inactive."""
    result = (
        get_supabase().table("api_keys")
        .select("id, active")
        .eq("key", x_api_key)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not result.data[0]["active"]:
        raise HTTPException(status_code=401, detail="API key has been revoked")

    return result.data[0]["id"]