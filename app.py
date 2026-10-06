import streamlit as st
import ollama
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# ChromaDB setup
# -----------------------------
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("college_faq")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🎓"
)

st.title("🎓 College FAQ Chatbot")
st.write("Ask me anything about the college!")


# -----------------------------
# Read college PDF
# -----------------------------
reader = PdfReader("documents/College Information.pdf")

pdf_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        pdf_text += text


# -----------------------------
# Read FAQs
# -----------------------------
with open("faqs.txt", "r", encoding="utf-8") as file:
    faqs = file.read().splitlines()


# -----------------------------
# User question
# -----------------------------
question = st.text_input("Ask your college question:")


if question:

    # Create question embedding
    query_embedding = embedding_model.encode(question).tolist()

    found = False

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    # -----------------------------
    # Get retrieved context safely
    # -----------------------------
    if results["documents"] and results["documents"][0]:
        context = "\n".join(results["documents"][0])
    else:
        context = "No relevant information found."

    # -----------------------------
    # Search FAQ file
    # -----------------------------
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

    # -----------------------------
    # If FAQ answer not found
    # use Ollama
    # -----------------------------
    if not found:

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Use this retrieved college information "
                        "to answer the question.\n\n"
                        f"{context}\n\n"
                        f"Question: {question}"
                    )
                }
            ]
        )

        st.write("🤖", response["message"]["content"])