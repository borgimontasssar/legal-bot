import os
import json
import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain import HuggingFaceHub
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter

# ─────────────────────────────────────────────
#              CONFIGURATION
# ─────────────────────────────────────────────

os.environ['QDRANT_HOST']         = "PASTE YOUR QDRANT HOST URL HERE"
os.environ['QDRANT_API_KEY']      = "PASTE YOUR QDRANT API KEY HERE"
os.environ['QDRANT_COLLECTION_NAME'] = "my-collection"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "PASTE YOUR HUGGINGFACE API TOKEN HERE"

# ─────────────────────────────────────────────
#              JSON FILES
# ─────────────────────────────────────────────

json_files = [
    "doc-2992.json",   # MAROUANE EL ABASSI
    "doc-3967.json",   # EL BEJI
    "doc-4093.json",   # Youssef El Chehed, Mohamed Anouar
    "doc-13675.json",  # MAROUANE EL ABASSI
    "doc-16617.json",  # Youssef Chahed
    "doc-19296.json",  # Elyes Fakhfakh
    "doc-19462.json",  # Mohamed Meherzi Abbou
    "doc-20626.json",  # من وزير المالية إلى
    "doc-24328.json",  # من وزير الاقتصاد والمالية
    "doc-38661.json",  # Marouane EL ABASSI
    "doc-39889.json",  # Yahia Chemlali
    "doc-57757.json",  # Najla Bouden Romdhane
]

# ─────────────────────────────────────────────
#         TEXT EXTRACTION FROM JSON
# ─────────────────────────────────────────────

output_list = []

def process_json_data(data):
    for item in data['result'].values():
        if isinstance(item, list):
            for nested_item in item:
                if not isinstance(nested_item, dict):
                    continue
                if 'name' in nested_item:
                    output_list.append(f"{nested_item['name']}")
                if 'published_date' in nested_item:
                    output_list.append(f" {nested_item['published_date']}")
                if 'fileURL' in nested_item:
                    output_list.append(f": {nested_item['fileURL']}")
                if 'contentMaillage' in nested_item and nested_item['contentMaillage'] is not None:
                    text = BeautifulSoup(nested_item['contentMaillage'], 'html.parser').get_text()
                    output_list.append(f"Content Maillage: {text}")

        elif isinstance(item, dict) and 'name' in item:
            output_list.append(f"{item['name']}")
            if 'published_date' in item:
                output_list.append(f" {item['published_date']}")
            if 'fileURL' in item:
                output_list.append(f"File URL: {item['fileURL']}")
            if 'contentMaillage' in item and item['contentMaillage'] is not None:
                text = BeautifulSoup(item['contentMaillage'], 'html.parser').get_text()
                output_list.append(f" {text}")


for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    process_json_data(data)

output_string = "\n".join(output_list)

# ─────────────────────────────────────────────
#         QDRANT VECTOR STORE SETUP
# ─────────────────────────────────────────────

client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_config = qdrant_client.http.models.VectorParams(
    size=768,  # 768 for paraphrase-multilingual-mpnet-base-v2
    distance=qdrant_client.http.models.Distance.COSINE
)

client.recreate_collection(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    vectors_config=collection_config
)

embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

vectorstore = Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings
)

# ─────────────────────────────────────────────
#         CHUNKING & INDEXING
# ─────────────────────────────────────────────

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

texts = get_chunks(output_string)
vectorstore.add_texts(texts)

# ─────────────────────────────────────────────
#         VECTOR STORE LOADER (for Streamlit)
# ─────────────────────────────────────────────

def get_vector_store():
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
    vectorstore = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings
    )
    return vectorstore

# ─────────────────────────────────────────────
#         STREAMLIT APP (Juri-bot)
# ─────────────────────────────────────────────

def truncate_text(text, max_tokens=2000):
    tokens = text.split(" ")
    return " ".join(tokens[:max_tokens])


def main():
    load_dotenv()
    st.set_page_config(page_title="Juri-bot")
    st.header("Posez Votre Question :")

    vectorstore = get_vector_store()

    qa = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-alpha",
            model_kwargs={"temperature": 0.2, "max_length": 512}
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    user_question = st.text_input("Question :")
    if user_question:
        truncated_question = truncate_text(user_question)
        st.write(f"Question: {truncated_question}")
        response = qa.run(truncated_question)
        st.write(f"Answer: {response}")


if __name__ == '__main__':
    main()
