import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection("college_faq")

model = SentenceTransformer("all-MiniLM-L6-v2")

reader_text = open("documents/college_info.txt", "r").read()

embedding = model.encode(reader_text).tolist()

collection.add(
    ids=["college_info"],
    documents=[reader_text],
    embeddings=[embedding]
)

print("Database created successfully!")