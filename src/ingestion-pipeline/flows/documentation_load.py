import os
import asyncio
import hashlib

from typing import Dict, List

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SitemapLoader
from langchain_core.documents import Document

from prefect import flow, task, unmapped
from prefect.cache_policies import NO_CACHE

from src.db.supabase.supabase_client import SupabaseManager
from src.db.qdrant.qdrant_client import QdrantStorage

from src.utils.logger import setup_logging

logger = setup_logging()

# Modified method from langchain docs to extract cleaner content 
def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")
    footer_elements = content.find_all("footer")

    # Remove each 'nav', 'header' and 'footer' element from the BeautifulSoup object
    for element in nav_elements + header_elements + footer_elements:
        element.decompose()


    text = content.get_text()
    
    # Remove "Skip to main content" anywhere in the text
    text = text.replace("Skip to main content", "")
    text = text.replace("On this page", "")
    
    # Remove "Last updated on" lines
    import re
    text = re.sub(r'Last updated on.*?\d{4}', '', text)
    
    # Remove footer text
    footer_start = text.find("Send us feedback")
    if footer_start != -1:
        text = text[:footer_start]
    
    # Clean up multiple newlines and extra spaces
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()

@task
def extract_title(document: Document) -> str:
    """
    Tries to extract the document page title in 3 ways
    
    1. From metadata 'title' field
    2. From the first line of the content
    3. From the URL

    Args:
        document (Document): The document object to extract the title from.

    Returns:
        str: The extracted title.
    """
    # using metadata
    title = document.metadata.get('title')
    if title:
        return title.split('|')[0].strip()
    
    # using the first line of the content
    content_lines = document.page_content.strip().split('\n')
    if content_lines and len(content_lines[0].strip()) > 5: 
        clean_title = content_lines[0].strip()

        if len(clean_title) < 100 and not any(keyword in clean_title for keyword in ["©", "Last updated"]):
             return clean_title
        
    # Using the URL
    url = document.metadata.get('source', '')
    url_title = os.path.basename(url.rstrip('/'))
    return url_title.replace('-', ' ').title()     
    

@task
async def load_documentation() -> List[Document]:
    """Loads all the documentation once"""
    loader = SitemapLoader(
        "https://docs.databricks.com/en/doc-sitemap.xml",
        filter_urls=["https://docs.databricks.com/aws/en/delta-sharing"],
        parsing_function=remove_nav_and_header_elements
    )
    docs = await asyncio.to_thread(loader.load) # Trick with asyncio to run sync code
    logger.info(f"{len(docs)} documents loaded.")
    return docs

@task
async def get_managers() -> tuple[SupabaseManager, QdrantStorage]:
    """Initializes and returns the Supabase and Qdrant managers."""
    vectorstore = QdrantStorage()
    await vectorstore.initialize()
    return SupabaseManager(), vectorstore

@task(cache_policy=NO_CACHE)
def get_source_id(db: SupabaseManager, source_name: str) -> str:
    """Obtains the UUID from the source 'Databricks Docs'.
    
    Args:
        db (SupabaseManager): The Supabase manager instance.
        source_name (str): The name of the source to look for.
    Returns:
        str: The UUID of the source.
    """
    source = db.get_source_by_name(source_name) 
    if not source:
        raise ValueError(f"Source '{source_name}' not found in the DB.")
    return source['source_id']

@task
def get_text_splitter(chunk_size=1500, chunk_overlap=200):
    """
    Returns a text splitter optimized for Databricks technical documentation.
    
    Uses hierarchical separators to preserve document structure, code blocks,
    and technical context. Prioritizes splitting at section headers, code blocks,
    and semantic boundaries rather than arbitrary character counts.
    
    Args:
        chunk_size: Maximum characters per chunk (default: 1500 for technical docs)
        chunk_overlap: Characters to overlap between chunks (default: 200 for context preservation)
    
    Returns:
        RecursiveCharacterTextSplitter configured for technical documentation
    """
    separators = [
        "\n## ",           # H2 headers - main sections (highest priority)
        "\n### ",          # H3 headers - subsections
        "\n#### ",         # H4 headers - sub-subsections
        "\n\n",            # Paragraph boundaries
        "\nPython",        # Python code blocks
        "\nSQL",           # SQL code blocks
        "\n```",           # Generic code blocks
        "\nnote",          # Important notes
        "\nwarning",       # Warnings
        "\nPreview",       # Preview features
        "\n- ",            # Bullet lists
        "\n* ",            # Alternative bullet lists
        "\n1. ",           # Numbered lists
        "\n",              # Single line breaks
        ". ",              # Sentence endings
        "; ",              # Semicolons (complex lists)
        ", ",              # Commas (last resort for lists)
        " ",               # Word boundaries
        "",                # Character-level fallback
    ]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

@task(retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
async def process_document(document, db: SupabaseManager, qdrant: QdrantStorage, source_id: str, text_splitter: RecursiveCharacterTextSplitter) -> Dict | None:
    """
    Processes a single document: checks for new or updated content, splits into chunks, and prepares metadata.

    Args:
        document (Document): The document to be processed.
        db (SupabaseManager): The Supabase manager instance.
        qdrant (QdrantStorage): The Qdrant storage instance.
        source_id (str): The source ID associated with the document.
        text_splitter (RecursiveCharacterTextSplitter): The text splitter instance.
    
    Returns:
        dict | None: A dictionary containing document ID, chunks, metadatas, action type, new hash, and number of chunks,
                      or None if the document is unchanged.
    """
    url = document.metadata.get('source')
    content = document.page_content

    if not content:
        logger.info(f"Omitted. empty document: {url}")
        return
    
    new_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

    existing_doc = await asyncio.to_thread(db.get_document_by_url, url)

    title = extract_title(document)

    if existing_doc is None:
        logger.info(f"New document: {url}")
        doc_id = await asyncio.to_thread(
            db.insert_document,
            source_id=source_id,
            url=url,
            title=title,
            doc_hash= new_hash
        )
        action = 'new'
        if not doc_id:
            return None
    elif existing_doc['hash'] != new_hash:
        #update
        logger.info("Updated document")
        doc_id = existing_doc['id']

        await qdrant.delete_by_document_id(doc_id)
        action = 'update'
    else:
        return None
    
    chunks = text_splitter.split_text(content)
    metadatas = [
        {
            "document_id": doc_id,
            "source_url": url,
            "title": title
        }
        for _ in chunks
    ]

    return {
        "doc_id": doc_id,
        "chunks": chunks,
        "metadatas": metadatas,
        "action": action,
        "new_hash": new_hash,
        "n_chunks": len(chunks)
    }

@task(cache_policy=NO_CACHE)
async def aggregate_and_ingest(results: List[Dict | None], qdrant: QdrantStorage, db: SupabaseManager):
    """
    Aggregates the processed document results and ingests new or updated chunks into Qdrant.
    Args:
        results (List[Dict | None]): List of processed document results.
        qdrant (QdrantStorage): The Qdrant storage instance.
        db (SupabaseManager): The Supabase manager instance.
    """
    all_chunks = []
    all_metadatas = []
    indicators = []

    for res in results:
        if res:
            all_chunks.extend(res["chunks"])
            all_metadatas.extend(res["metadatas"])
            indicators.append((res['doc_id'], res['action'], res['new_hash'], res['n_chunks']))

    if all_chunks:
        await qdrant.upsert(all_chunks, all_metadatas)
    else:
        logger.info("There are no new vectors to insert.")

    for doc_id, action, new_hash, n_chunks in indicators:
        if action == 'new':
            db.ingestion_checkpoint(doc_id, n_chunks)
        elif action == 'update':
            db.update_document_hash(doc_id, new_hash, n_chunks)   


@flow(name="Static Databricks Documentation Ingestion")
async def static_load_flow():
    """
    Flow to load and ingest Databricks documentation into the vector store:
        1. Loads documentation from the sitemap.
        2. Initializes database and vector store managers.
        3. Processes each document to check for new or updated content.
        4. Aggregates and ingests new or updated chunks into Qdrant.
        5. Updates the Supabase database with ingestion checkpoints.
    """
    db, qdrant = await get_managers()
    splitter = get_text_splitter()
    source_id = get_source_id(db, "Databricks Docs")

    docs = await load_documentation()

    logger.info("Mapping documents processing")
    results = process_document.map(
        document=docs,
        db=unmapped(db),
        qdrant=unmapped(qdrant),
        source_id=unmapped(source_id),
        text_splitter=unmapped(splitter)
    )
    
    await aggregate_and_ingest(results, qdrant, db)

if __name__ == "__main__":
    asyncio.run(static_load_flow())