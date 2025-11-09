import feedparser
from langchain_community.document_loaders.base import WebBaseLoader

def load_rss_documents(source, user_agent=None):
    """
    Load documents from an RSS feed.

    Args:
        source (str): The URL of the RSS feed.
        user_agent (str, optional): Custom User-Agent string for HTTP requests.

    Returns:
        List[Document]: A list of documents parsed from the RSS feed.
    """
    feed = feedparser.parse(source["rss_url"])
    docs = []
    headers = {"User-Agent": user_agent} if user_agent else {}

    for entry in feed.entries:
        url = entry.link
        title = entry.title
        try:
            loader = WebBaseLoader(url, requests_kwargs={"headers": headers})
            document = loader.load()[0]
            document.metadata["title"] = title
            document.metadata["source"] = url
            docs.append(document)
        except Exception as e:
            print(f"Error loading {url}: {e}")
    return docs