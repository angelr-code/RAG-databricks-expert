import asyncio

from typing import List

from langchain_community.document_loaders import SitemapLoader
from langchain_core.documents import Document

from prefect import flow, task, unmapped

from src.ingestion_pipeline.utils import (
    remove_nav_and_header_elements,
    get_managers,
    get_source_id,
    get_text_splitter,
    process_document,
    aggregate_and_ingest
)
from src.utils.logger import setup_logging

logger = setup_logging()

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
        text_splitter=unmapped(splitter),
        doc_type = unmapped('Documentation')
    )
    
    await aggregate_and_ingest(results, qdrant, db)

if __name__ == "__main__":
    asyncio.run(static_load_flow())