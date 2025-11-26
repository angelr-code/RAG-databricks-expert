import asyncio
import logging
from typing import List

import aiohttp
import feedparser

from prefect import task, flow, unmapped

from langchain_core.documents import Document

from src.utils.logger import setup_logging

from src.ingestion_pipeline.utils import (
    html_to_clean_text,
    get_managers,
    get_source_id,
    get_text_splitter,
    process_document,
    aggregate_and_ingest
)

logger = setup_logging()


@task(retries=2, retry_delay_seconds=5)
async def load_release_notes_feed(feed_url: str = "https://docs.databricks.com/aws/en/feed.xml") -> List[Document]:
    logger.info(f"Fetching Databricks AWS release notes: {feed_url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(feed_url) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Error fetching feed {feed_url}: {resp.status}")
                raise RuntimeError(f"Error fetching feed {feed_url}: {resp.status}")
            xml = await resp.text()
    
    docs: List[Document] = []
    feed = feedparser.parse(xml)
    for entry in feed.entries:
        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        published = getattr(entry, "published", "") or ""
        description_html = getattr(entry, "description", "") or ""

        clean_text = html_to_clean_text(description_html)

        page_content = f"{title}\n\nPublished: {published}\n\n{clean_text}"
        metadata = {
            "source": link,
            "title": title,
            "published": published,
            "feed_url": feed_url,
            "type": "release_note"
        }

        docs.append(Document(page_content=page_content, metadata=metadata))
    
    logger.info(f"Parsed {len(docs)} entries from feed.")
    return docs


@flow(name="Databricks Release Notes Ingestion - Historical + Daily")
async def release_notes_flow(feed_url: str = "https://docs.databricks.com/aws/en/feed.xml", run_type: str = "daily"):
    """
    Ingests Databricks Release Notes from the RSS feed into the vector database.

    Args:
        feed_url (str): The RSS feed URL for Databricks Release Notes.
        run_type (str): Type of run - "historical" for full ingestion,
                        "daily" for incremental ingestion.
    """
    db, qdrant = await get_managers()
    splitter = get_text_splitter()
    source_id = get_source_id(db, "Databricks Docs")

    docs = await load_release_notes_feed(feed_url=feed_url)

    if run_type == "historical":
        logger.info("Historical run requested. Will attempt to ingest all items; documents existing will still be compared by hash.")
        # Optionally force re-ingest ALL here.
        # WARNING: Uncomment only if you truly want to remove old vectors and rows:
        # existing_docs = db.get_documents_by_source_id(source_id)
        # for ed in existing_docs:
        #     await qdrant.delete_by_document_id(ed['id'])
        #     db.delete_document(ed['id'])

    logger.info("Mapping documents processing")
    BATCH_SIZE = 25
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        results = process_document.map(
            document=batch,
            db=unmapped(db),
            qdrant=unmapped(qdrant),
            source_id=unmapped(source_id),
            text_splitter=unmapped(splitter),
            doc_type = unmapped('Release Notes')
        )
        await aggregate_and_ingest(results, qdrant, db)
        logger.info(f"Release Notes {i} to {i + BATCH_SIZE} ingested in Qdrant")
        await asyncio.sleep(1)

    logger.info("Release Notes ingestion finished")

    logger.info("Release Notes ingestion finished")

if __name__ == '__main__':
    """
    To run daily ingestion:
        asyncio.run(release_notes_flow())
    To run historical ingestion:
        asyncio.run(release_notes_flow(run_type="historical"))
    """
    asyncio.run(release_notes_flow())