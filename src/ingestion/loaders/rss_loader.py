from langchain_community.document_loaders import RSSFeedLoader

def load_rss_documents(source: dict):
    """Loads documents from a RSS feed"""
    print(f"Loading RSS from: {source['base_url']}")
    try:
        loader = RSSFeedLoader(urls=[source['base_url']], )
        docs = loader.load()
        # We add the necessary metadata for posterior processing
        for doc in docs:
            doc.metadata['source_name'] = source['name']
            doc.metadata['source_type'] = source['type']
            doc.metadata['unique_id'] = doc.metadata.get('guid', doc.metadata.get('link'))
    except Exception as e:
        print(f"Error loading {source['base_url']}: {e}")
        return []