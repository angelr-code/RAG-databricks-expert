import os 
from supabase import create_client
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

from src.utils.logger import setup_logging

from typing import Optional, Dict, Any, List

logger = setup_logging()

class SupabaseManager:
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.sources_table = 'sources'
        self.documents_table = 'documents'

    # --- Sources management ---

    def insert_source(self, name: str, base_url: str, source_type: str) -> Optional[Dict[str, Any]]:
        """
        Inserts a new source into the sources table.
        
        Args:
            name (str): The name of the source.
            base_url (str): The base URL of the source.
            type (str): The type of the source. This is, 'static' or 'dynamic'.
        Returns:
            Optional[Dict[str, Any]]: The inserted source record or None if insertion failed.
        """
        try:
            data = {
                'name': name,
                'base_url': base_url,
                'type': source_type,
            } 
            response = self.client.table(self.sources_table)\
                                .insert(data)\
                                .execute()
            if response.data:
                logger.info(f"New source created: {name}")
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error inserting a new source: {e}")
            return None
        
    def get_source_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a source by its name.

        Args:
            name (str): The name of the source to retrieve.
        Returns:
            Optional[Dict[str, Any]]: The source record if found, otherwise None.
        """
        try:
            response = self.client.table(self.sources_table)\
                                .select("*")\
                                .eq("name", name)\
                                .execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error finding {name}: {e}")
            return None
        
    def list_sources(self) -> List:
        """
        Lists all the registered sources
        """
        try:
            response = self.client.table(self.sources_table)\
                                .select("*")\
                                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            return []

        
    # --- Documents management ---

    def get_document_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Searchs for a document based on its URL (used to check for updates)
        
        Args:
            url (str): The URL of the document to search for.
        Returns:
            Optional[Dict[str, Any]]: The document record if found, otherwise None.
        """
        try:
            response = self.client.table(self.documents_table)\
                                .select("id, hash")\
                                .eq("url", url)\
                                .execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error checking the document for url {url}: {e}")
            return None
        
    def update_document_hash(self, document_id: str, new_hash: str, n_chunks: int) -> bool:
        """
        Updates the hash of a document when its content has been ingested to Qdrant
        Args:
            document_id (str): The ID of the document to update.
            new_hash (str): The new hash value.
            n_chunks (int): The number of chunks the document was split into.
        Returns:
            bool: True if the update was successful, False otherwise.
        """
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
                logger.info(f"Document {document_id} updated with new hash.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating document {document_id} hash: {e}")
            return False
        
    def ingestion_checkpoint(self, document_id: str, n_chunks: int) -> bool:
        """
        Updates the ingested_at and n_chunks variables for documents already ingested in Qdrant
        
        Args:
            document_id (str): The ID of the document to update.
            n_chunks (int): The number of chunks the document was split into.
        Returns:
            bool: True if the update was successful, False otherwise.
        """
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
                logger.info(f"Document {document_id} updated.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
                
    def insert_document(self, source_id: str, url: str, doc_hash: str, title: str) -> Optional[str]:
        """"
        Inserts a new document into the documents table.

        Args:
            source_id (str): The ID of the source the document belongs to.
            url (str): The URL of the document.
            doc_hash (str): The hash of the document content.
            title (str): The title of the document.
        Returns:
            Optional[str]: The ID of the inserted document or None if insertion failed.
        """
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
                logger.info(f"Document created (pending vectorial insertion): {doc_id} | {url}")
                return doc_id
            return None           
        except Exception as e:
           logger.error(f"Error inserting document {doc_hash}: {e}")
           return None
