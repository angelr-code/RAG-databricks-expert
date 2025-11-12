import os 
from supabase import create_client
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL", "http://127.0.0.1:54321")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_secret_N7UND0UgjKTVK-Uodkm0Hg_xSvEMPvz")

class SupabaseManager:
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.sources_table = 'sources'
        self.documents_table = 'documents'

    # --- Sources management ---

    def insert_source(self, name, base_url, type):
        try:
            data = {
                'name': name,
                'base_url': base_url,
                'type': type,
            }
            response = self.client.table(self.sources_table)\
                                .insert(data)\
                                .select("*")\
                                .execute()
            if response.data:
                print(f"New source created: {name}")
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error inserting a new source: {e}")
            return None
        
    def get_source_by_name(self, name):
        try:
            response = self.client.table(self.sources_table)\
                                .select("*")\
                                .eq("name", name)\
                                .execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error finding {name}: {e}")
            return None
        
    def list_sources(self):
        """Lists all the registered sources"""
        try:
            response = self.client.table(self.sources_table)\
                                .select("*")\
                                .execute()
            return response.data
        except Exception as e:
            print(f"Error listing sources: {e}")
            return []

        

    # --- Documents management ---

    def get_document_by_url(self, url):
        """Searchs for a document based on its URL (to check for updates)"""
        try:
            response = self.client.table(self.documents_table)\
                                .select("id, hash")\
                                .eq("url", url)\
                                .execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error checking the document for url {url}: {e}")
            return None
        
    def update_document_hash(self, document_id, new_hash, n_chunks):
        """Updates the hash of a document when its content has been ingested to Qdrant"""
        try:
            data = {
                "hash": new_hash,
                "ingested_at": datetime.now().isoformat(),
                "n_chunks": n_chunks
            }
            response = self.client.table(self.documents_table)\
                                .update(data)\
                                .eq("id", document_id)\
                                .execute()
            if response.data:
                print(f"Document {document_id} updated with new hash.")
                return True
            return False
        except Exception as e:
            print(f"Error updating document {document_id} hash: {e}")
            return False
        
    def ingestion_checkpoint(self, document_id, n_chunks):
        """Updates the ingested_at and n_chunks variables for documents already ingested in Qdrant"""
        try:
            data = {
                "ingested_at": datetime.now().isoformat(),
                "n_chunks": n_chunks
            }
            response = self.client.table(self.documents_table)\
                                .update(data)\
                                .eq("id", document_id)\
                                .execute()
            if response.data:
                print(f"Document {document_id} updated.")
                return True
            return False
        except Exception as e:
            print(f"Error updating document {document_id}: {e}")
            return False
                
    def insert_document(self, source_id, url, doc_hash, title):
        try:
            data = {
                "source_id": source_id,
                "url": url,
                "hash": doc_hash,
                "title": title,
            } 
            response = self.client.table(self.documents_table)\
                                .insert(data)\
                                .execute()
            if response.data:
                doc_id = response.data[0]['id']
                print(f"Document created (pending vectorial insertion): {doc_id} | {url}")
                return doc_id
            return None           
        except Exception as e:
           print(f"Error inserting document {doc_hash}: {e}")
           return None

