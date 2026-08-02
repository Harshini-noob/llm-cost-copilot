import os
from dotenv import load_dotenv
from supabase import create_client
from fastapi import Header, HTTPException

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def verify_api_key(x_api_key: str = Header(...)):
    """FastAPI dependency — runs before the endpoint body executes.
    Rejects the request early if the key is missing, unknown, or inactive."""
    result = (
        supabase.table("api_keys")
        .select("id, active")
        .eq("key", x_api_key)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not result.data[0]["active"]:
        raise HTTPException(status_code=401, detail="API key has been revoked")

    return result.data[0]["id"]  # available to the endpoint if needed later