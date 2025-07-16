from langchain_community.document_loaders import WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

import requests
from urllib.parse import urlparse
import time

def load_and_split_urls(urls):
    """Load and split documents from URLs with robust error handling"""
    
    if not urls:
        print("No URLs provided")
        return []
    
    documents = []
    
    for url in urls:
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                print(f"Invalid URL format: {url}")
                continue
            
            print(f"Loading URL: {url}")
            
            # Use WebBaseLoader instead of UnstructuredURLLoader
            loader = WebBaseLoader(
                web_paths=[url],
                header_template={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )
            
            # Load with timeout handling
            docs = loader.load()
            
            if docs and len(docs) > 0:
                # Filter out empty documents
                valid_docs = [doc for doc in docs if doc.page_content.strip()]
                if valid_docs:
                    documents.extend(valid_docs)
                    print(f"✅ Loaded {len(valid_docs)} valid documents from {url}")
                else:
                    print(f"⚠️ No valid content found in {url}")
            else:
                print(f"⚠️ No documents loaded from {url}")
                
            # Add small delay to be respectful to servers
            time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error loading {url}: {str(e)}")
            continue
    
    if not documents:
        print("❌ No documents were successfully loaded from any URL")
        return []
    
    # Split documents
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Filter out very short chunks
        valid_chunks = [doc for doc in split_docs if len(doc.page_content.strip()) > 50]
        
        print(f"✅ Created {len(valid_chunks)} valid document chunks")
        
        return valid_chunks
        
    except Exception as e:
        print(f"❌ Error splitting documents: {str(e)}")
        return []


# Alternative fallback function using requests + BeautifulSoup
def load_and_split_urls_fallback(urls):
    """Fallback method using requests and BeautifulSoup"""
    
    import requests
    from bs4 import BeautifulSoup
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    documents = []
    
    for url in urls:
        try:
            print(f"Loading URL (fallback): {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if text and len(text.strip()) > 100:
                doc = Document(page_content=text, metadata={"source": url})
                documents.append(doc)
                print(f"✅ Loaded content from {url} (fallback)")
            else:
                print(f"⚠️ No sufficient content found in {url}")
                
        except Exception as e:
            print(f"❌ Error loading {url} (fallback): {str(e)}")
            continue
    
    if not documents:
        return []
    
    # Split documents
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        split_docs = text_splitter.split_documents(documents)
        valid_chunks = [doc for doc in split_docs if len(doc.page_content.strip()) > 50]
        
        print(f"✅ Created {len(valid_chunks)} valid chunks (fallback)")
        return valid_chunks
        
    except Exception as e:
        print(f"❌ Error splitting documents (fallback): {str(e)}")
        return []
