import streamlit as st
import os
import pickle
from utils.ingestion import load_and_split_urls
from utils.embeddings import get_hf_embeddings
from utils.llm_interface import get_local_llm
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


st.set_page_config(layout="wide")
st.title("🧠 RockyBot Pro: Local News Research Assistant")


if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
    st.success("Chat history cleared.")


# Sidebar input
st.sidebar.header("Enter News URLs")
raw_urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
urls = [url for url in raw_urls if url.strip()]
process = st.sidebar.button("🔍 Process URLs")

faiss_file = "vectorstore/faiss_store.pkl"
placeholder = st.empty()

if process:
    try:
        st.write(f"🔄 Processing {len(urls)} URLs...")
        st.write(f"URLs: {urls}")
        
        # Debug: Check document loading
        docs = load_and_split_urls(urls)
        st.write(f"✅ Loaded {len(docs)} documents")
        
        if len(docs) == 0:
            st.error("❌ No documents were loaded. Check your URLs and ingestion function.")
        else:
            # Show first few characters of first document
            st.write(f"📄 First doc preview: {docs[0].page_content[:200]}...")
            
            # Debug: Check embeddings
            st.write("🔄 Creating embeddings...")
            embeddings = get_hf_embeddings()
            st.write("✅ Embeddings model loaded")
            
            # Debug: Check FAISS creation
            st.write("🔄 Creating FAISS store...")
            db = FAISS.from_documents(docs, embeddings)
            st.write("✅ FAISS store created")
            
            # Save to file
            os.makedirs("vectorstore", exist_ok=True)
            with open(faiss_file, "wb") as f:
                pickle.dump(db, f)
            placeholder.success("✅ FAISS store saved!")
            
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

# Query input
query = st.text_input("Ask a question from the articles:")
if query:
    if os.path.exists(faiss_file):
        try:
            with open(faiss_file, "rb") as f:
                db = pickle.load(f)
            
            llm = get_local_llm()
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(),
                memory=memory,
                return_source_documents=True,
                output_key = 'answer'
            )
            result = chain.invoke({"question": query})
            st.session_state.chat_history.append(("User", query))
            st.session_state.chat_history.append(("RockyBot", result["answer"]))
            
            st.subheader("Answer:")
            st.write(result["answer"])
            
            st.subheader("Sources:")
            source_docs = result.get("source_documents", [])
            if source_docs:
                for doc in source_docs:
                    st.write(f"- {doc.metadata.get('source', 'Unknown')}")
            else:
                st.write("No sources found.")
                
        except Exception as e:
            st.error(f"❌ Error during query: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

    else:
        st.warning("⚠️ Please process some URLs first.")


with st.expander("🕒 Chat History", expanded=False):
    if st.session_state.chat_history:
        for sender, message in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {message}")
    else:
        st.write("No conversation yet.")

