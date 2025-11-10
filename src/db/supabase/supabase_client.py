from supabase import create_client
import os 
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def supabase_client():
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # =============================
    # Sources
    # =============================
    def insert_source(self, source: dict):
        res = self.client.table("sources").insert(source).execute()
        return res.data[0]

    def get_active_sources(self):
        res = self.client.table("sources").select("*").eq("active", True).execute()
        return res.data
    
    # =============================
    # Documents
    # =============================
    
    def exists_document(self, hash_value: str):
        res = self.client.table("documents").select("id").eq("hash", hash_value).execute()
        return len(res.data) > 0
    
    def insert_document(self, data: dict):
        res = self.client.table("documents").insert(data).execute()
        return res.data[0]
    
    def update_document_status(self, document_id, status, n_chunks = None):
        update_data = {"status": status, "updated_at": datetime.now().isoformat()}
        if n_chunks is not None:
            update_data["n_chunks"] = n_chunks
        self.client.table("documents").update(update_data).eq("id", document_id).execute()
