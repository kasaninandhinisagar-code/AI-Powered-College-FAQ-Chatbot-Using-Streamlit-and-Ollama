import streamlit as st
import ollama
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("college_faq")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="College FAQ Chatbot", page_icon="🎓")

st.title("🎓 College FAQ Chatbot")
st.write("Ask me anything about the college!")

reader = PdfReader("documents/College Information.pdf")

pdf_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text

with open("faqs.txt", "r") as file:
    faqs = file.read().splitlines()

question = st.text_input("Ask your college question:")
query_embedding = embedding_model.encode(question).tolist()

if question:
    query_embedding = embedding_model.encode(question).tolist()
    found = False
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    context = results["documents"][0][0]

    for faq in faqs:
        if ":" in faq:
            keys, answer = faq.split(":", 1)

            for key in keys.split(","):
                if key.strip().lower() in question.lower():
                    st.write("🤖", answer.strip())
                    found = True
                    break

        if found:
            break

    if not found:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": f"Use this retrieved college information to answer the question:\n\n{context}\n\nQuestion: {question}"
                }
            ]
        )

        st.write("🤖", response["message"]["content"])