import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

result = supabase.table("requests").insert({
    "prompt": "What is 2+2?",
    "tier": "simple",
    "routing_mode": "balanced"
}).execute()

print(result)