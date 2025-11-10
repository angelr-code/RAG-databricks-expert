from langchain_community.document_loaders import ArxivLoader

def load_arxiv_papers(source: dict):
    query = source["base_url"]
    print(f"Loading Arxiv papers: {query}")
    try:
        loader = ArxivLoader(query)
        docs = loader.load()
        # We add the necessary metadata for posterior processing
        for doc in docs:
            doc.metadata['source_name'] = source['name']
            doc.metadata['source_type'] = source['type']
            doc.metadata['unique_id'] = doc.metadata.get('entry_id')
            doc.page_content = doc.metadata.get('Summary', 'No summary available.')

        return docs
    except Exception as e:
        print(f"Error loading arxiv papers: {e}")
