import sys
import os
import uuid
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import CreateCollection, VectorParams, Distance, PointStruct

SOURCE_URL = "https://www.deeplearning.ai/the-batch/check-out-our-course-on-how-to-build-ai-agents/"
USER_AGENT = os.environ.get("USER_AGENT")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "docs"


load_dotenv()

class QdrantStorage():
    def __init__(self, url=QDRANT_URL, collection=QDRANT_COLLECTION, dim=EMBEDDING_DIM):
        self.client = QdrantClient(url, timeout=15)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id = ids[i], vector = vectors[i], payload = payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points = points)

    def search(self, query_vector, top_k = 5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set() # set in order to not repeat sources

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if source:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}


print(f"Loading embedding model: {MODEL_NAME}")
embeddings = HuggingFaceEmbeddings(model_name= MODEL_NAME)
print("Model loaded.\n")


print(f"Loading document from: {SOURCE_URL}")
CUSTOM_HEADERS = {
    'User-Agent': USER_AGENT,
}
loader = WebBaseLoader(
    SOURCE_URL, 
    requests_kwargs={"headers": CUSTOM_HEADERS}
)
docs = loader.load()
print(f"Document loaded. Title: {docs[0].metadata['title']}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 200
)

splits = text_splitter.split_documents(docs) 
print(f"Document divided into {len(splits)} chunks.\n")

#DB LOADING

qdrant = QdrantStorage()
texts_to_embbed = [chunk.page_content for chunk in splits]
vectors = embeddings.embed_documents(texts_to_embbed)
ids = [str(uuid.uuid4()) for _ in splits]

payloads = []
for chunk in splits:
    payloads.append({
        "text": chunk.page_content,
        "source": chunk.metadata.get("source", SOURCE_URL),
        "title": chunk.metadata.get("title", "N/A")
    })

print("Ingesting documents into Qdrant...")
qdrant.upsert(ids=ids, vectors=vectors, payloads=payloads)
print("Successful ingestion\n")

query = "What are the four design patterns teached in the course?"
query_vector = embeddings.embed_documents([query])[0]

result = qdrant.search(query_vector, top_k=2)
print(result["contexts"])

