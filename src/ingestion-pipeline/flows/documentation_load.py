import os
import uuid
import hashlib
import asyncio
from bs4 import BeautifulSoup
from prefect import flow, task, unmapped
from prefect.cache_policies import NO_CACHE
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SitemapLoader
from langchain_huggingface import HuggingFaceEmbeddings

from src.db.supabase.supabase_client import SupabaseManager
from src.db.qdrant.qdrant_client import QdrantStorage

# Method from langchain docs to extract cleaner content 
def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")

    # Remove each 'nav' and 'header' element from the BeautifulSoup object
    for element in nav_elements + header_elements:
        element.decompose()

    return str(content.get_text())

@task
def extract_title(document):
    """Tries to extract the document page title in 3 ways"""
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
async def load_documentation():
    """Loads all the documentation once"""
    loader = SitemapLoader(
        "https://docs.databricks.com/en/doc-sitemap.xml",
        filter_urls=["https://docs.databricks.com/aws/en/delta-sharing"],
        parsing_function=remove_nav_and_header_elements
    )
    docs = await asyncio.to_thread(loader.load) # Trick with asyncio to run sync code
    print(f"{len(docs)} documents loaded.")
    return docs

@task
async def get_managers():
    vectorstore = QdrantStorage()
    await vectorstore.initialize()
    return SupabaseManager(), vectorstore

@task(cache_policy=NO_CACHE)
def get_source_id(db: SupabaseManager, source_name: str) -> str:
    """Obtains the UUID from the source 'Databricks Docs'."""
    source = db.get_source_by_name(source_name) 
    if not source:
        raise ValueError(f"Source '{source_name}' not found in the DB.")
    return source['source_id']

@task
def get_text_splitter(chunk_size = 1000, chunk_overlap = 200):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

@task 
def get_embedding_model(model_name = "all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

@task(retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
async def process_document(document, db: SupabaseManager, qdrant: QdrantStorage, source_id: str, text_splitter: RecursiveCharacterTextSplitter, embedding_model: HuggingFaceEmbeddings):
    """Saves documents metadata in Supabase and prepares for Qdrant ingestion"""
    url = document.metadata.get('source')
    content = document.page_content

    if not content:
        print(f"Omitted. empty document: {url}")
        return
    
    new_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

    existing_doc = await asyncio.to_thread(db.get_document_by_url, url)

    title = extract_title(document)

    if existing_doc is None:
        print(f"New document: {url}")
        doc_id = await asyncio.to_thread(
            db.insert_document,
            source_id=source_id,
            url=url,
            title=title,
            doc_hash= new_hash
        )
        status = 'new'
        if not doc_id:
            return None
    elif existing_doc['hash'] != new_hash:
        #update
        print("Updated document")
        doc_id = existing_doc['id']

        await qdrant.delete_by_document_id(doc_id)
        status = 'update'
    else:
        return None
    
    chunks = text_splitter.split_text(content)
    vectors = asyncio.to_thread(embedding_model.embed_documents, chunks)
    if len(chunks) != len(vectors) or not vectors:
        print(f"[ERROR] Disparidad en chunks/vectores para {url}. Chunks: {len(chunks)}, Vectores: {len(vectors)}")
        return None
    ids = [str(uuid.uuid4()) for _ in chunks]
    payloads = [
        {
            "document_id": doc_id,
            "source_url": url,
            "text": chunk,
            "title": title
        }
        for chunk in chunks
    ]

    return {
        "doc_id": doc_id,
        "ids": ids,
        "vectors": vectors,
        "payloads": payloads,
        "status": status,
        "new_hash": new_hash,
        "n_chunks": len(ids)
    }

@task(cache_policy=NO_CACHE)
async def aggregate_and_ingest(results: list[dict | None], qdrant: QdrantStorage, db: SupabaseManager) -> dict:
    all_ids = []
    all_vectors = []
    all_payloads = []
    indicators = []

    for res in results:
        if res:
            all_ids.extend(res["ids"])
            all_vectors.extend(res["vectors"])
            all_payloads.extend(res["payloads"])
            indicators.append((res['doc_id'], res['status'], res['new_hash'], res['n_chunks']))

    if all_ids:
        await qdrant.upsert(all_ids, all_vectors, all_payloads)
    else:
        print("There are no new vectors to insert.")

    for doc_id, action, new_hash, n_chunks in indicators:
        if action == 'new':
            db.ingestion_checkpoint(doc_id, n_chunks)
        elif action == 'update':
            db.update_document_hash(doc_id, new_hash, n_chunks)   


@flow(name="Static Databricks Documentation Ingestion")
async def static_load_flow():
    db, qdrant = await get_managers()
    splitter = get_text_splitter()
    model = get_embedding_model()
    source_id = get_source_id(db, "Databricks Docs")

    docs = await load_documentation()

    print("Mapping documents processing")
    results = process_document.map(
        document=docs,
        db=unmapped(db),
        qdrant=unmapped(qdrant),
        source_id=unmapped(source_id),
        text_splitter=unmapped(splitter),
        embedding_model=unmapped(model)
    )
    
    await aggregate_and_ingest(results, qdrant, db)


if __name__ == "__main__":
    asyncio.run(static_load_flow())