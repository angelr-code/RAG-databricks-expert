import os 
from supabase import create_client
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class SupabaseManager:
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.sources_table = 'sources'
        self.documents_table = 'documents'

    # --- Sources management ---

    def insert_source(self, name, base_url, type, description, active):
        try:
            data = {
                'name': name,
                'base_url': base_url,
                'type': type,
                'description': description,
                'active': active
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
        
    def update_source_status(self, source_id, active):
        """Updates source status to either active or inactive"""
        try:
            response = self.client.table("sources")\
                                .update({"active": active, "updated_at": datetime.now().isoformat()})\
                                .eq("id", source_id)\
                                .update("*")\
                                .execute()
            if response.data:
                print(f"Source {id} status changed to active={active}")
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error modifying the source {id} status: {e}")
            return None
        
    def list_sources(self):
        """Lists all the registered sources"""
        try:
            response = self.client.table("sources")\
                                .select("*")\
                                .execute()
            return response.data
        except Exception as e:
            print(f"Error listing sources: {e}")
            return []

    def get_active_sources(self):
        """Returns all the sources labeled as active"""
        try:
            response = self.client.table("sources")\
                                .select("*")\
                                .eq("active", True)\
                                .execute()
            return response.data
        except Exception as e:
            print(f"Error fetching active sources: {e}")
            return []
        

    # --- Documents management ---

    def get_document_by_hash(self, hash):
        """Searchs for a document based on its hash"""
        try:
            response = self.client.table("documents")\
                                .select("id")\
                                .eq("hash", hash)\
                                .execute()
            return response.data
        except Exception as e:
            print(f"Error checking the hash {hash}: {e}")
            return None
        
    def insert_document(self, source_id, source_url, doc_hash, metadata):
        try:
            data = {
                "source_id": source_id,
                "source_url": source_url,
                "hash": doc_hash,
                "title": metadata.get('title'),
                "author": metadata.get('author'),
                "published_at": metadata.get('published'),
                "summary": metadata.get('summary'),
                "status": "pending",
                "meta": metadata
            } 
            response = self.client.table(self.documents_table)\
                                .insert(data)\
                                .select("id")\
                                .execute()
            if response.data:
                doc_id = response.data[0]['id']
                print(f"Document created (pending): {doc_id} | {source_url}")
                return doc_id
            return None           
        except Exception as e:
           print(f"Error inserting docuement {doc_hash}: {e}")
           return None

    def update_source_status(self, doc_id, status, n_chunks = 0):
        """Updates source status to either active or inactive"""
        update_data = {
            'status': status,
            'updated_at': datetime.now().isoformat()
        }
        if status == "indexed":
            update_data['n_chunks'] = n_chunks
        
        try:
            self.client.table(self.documents_table)\
                        .update(update_data)\
                        .equal("id", doc_id)\
                        .execute()
            print(f"Document status updated: {doc_id} -> {status}")
        except Exception as e:
            print(f"Error updating document {doc_id} status: {e}")
