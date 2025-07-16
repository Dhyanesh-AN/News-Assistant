# 🗞️ News-Assistant  
**AI-Powered News Article Explainer**

- 🔗 Ingest articles from **URLs**
- 🧩 Intelligent document **chunking** and **embedding** using HuggingFace models
- 📚 **FAISS**-based vector similarity search for relevant context retrieval
- 🤖 Local **LLM interaction** using [Ollama](https://ollama.com/)
- 🧠 Supports **chat memory** and **follow-up questions**
- 💡 Real-time explanations, summaries, and Q&A from long-form news articles
- 🌐 Fully **offline** and **privacy-friendly** – No external API calls required

---

## 🧪 Steps To Run

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Run a Local Model via Ollama**

```bash
ollama run llama3
```

> ℹ️ To use a different model, update `model="llama3"` in `app.py`.

3. **Launch the Streamlit App**

```bash
streamlit run app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501)

---
