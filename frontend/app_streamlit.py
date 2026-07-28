import os
import streamlit as st
import requests

# Define API URL, fallback to localhost if not in docker
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1/search")

st.set_page_config(
    page_title="Book Semantic Search V2",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Semantic Search Engine V2")
st.markdown("""
Welcome to the production-grade Book Search Engine. 
This engine uses **Hybrid Search (BM25 + FAISS)** combined with **Cross-Encoder Reranking** for maximum accuracy.
""")

query = st.text_input("Search for a book (e.g., 'Kisah cinta masa penjajahan', 'Cara membangun kebiasaan'):", "")

col1, col2 = st.columns([1, 5])
with col1:
    top_k = st.number_input("Top K Results", min_value=1, max_value=20, value=5)

if st.button("Search", type="primary"):
    if query:
        with st.spinner("Searching and Reranking..."):
            try:
                response = requests.get(
                    API_URL,
                    params={"q": query, "top_k": top_k},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    latency = data.get("latency_ms", 0.0)
                    
                    st.success(f"Found {len(results)} results in {latency:.2f} ms")
                    
                    if not results:
                        st.info("No matching books found.")
                    
                    for idx, res in enumerate(results):
                        with st.container():
                            st.subheader(f"#{idx+1} - {res['title']}")
                            st.caption(f"**Author**: {res['author']} | **Genre**: {res['genre']} | **Relevance Score**: {res['score']:.4f}")
                            st.write(res['synopsis'])
                            st.divider()
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Failed to connect to the backend API at {API_URL}. Is it running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a search query.")
