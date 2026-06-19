# ⚖️ Legal-Bot — Multilingual Legal Q&A Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers legal questions in **Arabic and French** by retrieving context from structured **JSON files**. Built with LangChain, Qdrant, HuggingFace, and Streamlit.

---

## 🧠 How It Works

1. Legal documents are loaded from **structured JSON files**
2. Text is extracted and cleaned using **BeautifulSoup**
3. Text is split into chunks and embedded using a **multilingual sentence transformer**
4. Embeddings are stored in a **Qdrant vector database**
5. User asks a question → relevant chunks are retrieved → **Zephyr-7B LLM** generates an answer

```
JSON Files → Text Extraction → Chunking → Embeddings → Qdrant
                                                           ↓
                                User Question → Retriever → LLM → Answer
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [LangChain](https://www.langchain.com/) | RAG pipeline & LLM chaining |
| [Qdrant](https://qdrant.tech/) | Vector database for embeddings |
| [HuggingFace](https://huggingface.co/) | LLM (Zephyr-7B) + embeddings |
| [Sentence Transformers](https://www.sbert.net/) | `paraphrase-multilingual-mpnet-base-v2` |
| [Streamlit](https://streamlit.io/) | Web UI |
| [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) | HTML content parsing from JSON |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/borgimontasssar/legal-bot.git
cd legal-bot
```

### 2. Install dependencies

```bash
pip install langchain qdrant-client sentence-transformers streamlit beautifulsoup4 python-dotenv huggingface_hub
```

### 3. Get your API keys

**Qdrant (Vector Database):**
- Go to [https://cloud.qdrant.io](https://cloud.qdrant.io) and create a free account
- Create a new **Cluster**
- Copy your **Cluster URL** and **API Key** from the dashboard

**HuggingFace:**
- Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Click **"New token"** → give it a name → select **"Read"** role
- Copy the token

### 4. Add your API keys to the script

Open `legal_bot.py` and fill in the placeholders at the top of the file:

```python
os.environ['QDRANT_HOST']            = "PASTE YOUR QDRANT HOST URL HERE"
os.environ['QDRANT_API_KEY']         = "PASTE YOUR QDRANT API KEY HERE"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "PASTE YOUR HUGGINGFACE API TOKEN HERE"
```

### 5. Set up the Qdrant collection

The script automatically creates a Qdrant collection called `my-collection` with:
- **Vector size:** 768 (matching `paraphrase-multilingual-mpnet-base-v2`)
- **Distance metric:** Cosine similarity

No manual setup needed — it runs on first launch.

### 6. Add your JSON files

Place your JSON files in the same directory as `legal_bot.py`. The expected structure for each JSON file is:

```json
{
  "result": {
    "key": [
      {
        "name": "Document title",
        "published_date": "2023-01-01",
        "fileURL": "https://example.com/doc.pdf",
        "contentMaillage": "<p>HTML content here...</p>"
      }
    ]
  }
}
```

Update the `json_files` list in the script with your filenames:

```python
json_files = [
    "your-file-1.json",
    "your-file-2.json",
    ...
]
```

---

## 🚀 Running the App

### Option 1 — Streamlit Web UI (recommended)

```bash
streamlit run legal_bot.py
```

Then open your browser at `http://localhost:8501`

### Option 2 — Command Line

Uncomment the CLI section at the bottom of the script and run:

```bash
python legal_bot.py
```

Type your question in Arabic or French and press Enter. Type `exit` to quit.

---

## 📁 Project Structure

```
legal-bot/
│
├── legal_bot.py        # Main script (indexing + Streamlit app)
├── README.md           # This file
└── *.json              # Your legal document JSON files (not included)
```

---

## 💡 Key Features

- **Multilingual** — supports Arabic and French out of the box
- **JSON-native RAG** — retrieves directly from structured JSON, not PDFs or plain text
- **Streamlit UI** — clean web interface for non-technical users
- **Scalable** — add more JSON files to expand the knowledge base

---

## 📌 Notes

- The JSON files referenced in this repo contain sensitive legal documents and are **not included** in the repository
- Make sure to **never commit your API keys** — use environment variables or a `.env` file in production
- For production use, consider moving API keys to a `.env` file and using `python-dotenv`

---

## 👤 Author

**Montassar Borgi**
- GitHub: [@borgimontasssar](https://github.com/borgimontasssar)
