import streamlit as st
import os
import sys
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model_loader import ModelLoader
from src.vector_indexer import Indexer
from src.ranker import Ranker

# Page Config
st.set_page_config(
    page_title="Vision Scout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
        padding: 1rem;
        border-radius: 10px;
    }
    .image-card {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .image-card:hover {
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_components():
    return ModelLoader(), Indexer(), Ranker()

def main():
    st.title("🔍 Vision Scout")
    st.markdown("### Zero-Shot Semantic Image Search")
    
    # Load components
    try:
        model_loader, indexer, ranker = load_components()
    except Exception as e:
        st.error(f"Error loading components: {e}")
        st.stop()
        
    # Load descriptions for re-ranking
    @st.cache_resource
    def load_descriptions():
        csv_path = os.path.join(os.path.dirname(__file__), 'assets/unsplash-research-dataset-lite-latest/photos.csv000')
        desc_map = {}
        if os.path.exists(csv_path):
            import csv
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        desc = row.get('ai_description') or row.get('photo_description')
                        if desc:
                            desc_map[row['photo_id']] = desc
            except Exception as e:
                st.error(f"Error loading descriptions: {e}")
        return desc_map

    desc_map = load_descriptions()
        
    # Search Bar
    query = st.text_input("Describe what you're looking for...", placeholder="e.g., 'a futuristic city at night' or 'a happy dog running'")
    
    if query:
        with st.spinner("Searching..."):
            # Generate text embedding
            text_embedding = model_loader.get_text_embedding(query)
            
            if text_embedding:
                # Search Pinecone (Fetch top 50 for re-ranking)
                results = indexer.search(text_embedding, top_k=50)
                
                if results and results['matches']:
                    # Prepare candidates for re-ranking
                    candidates = []
                    for match in results['matches']:
                        filename = match['metadata'].get('filename', '')
                        pid = os.path.splitext(filename)[0]
                        description = desc_map.get(pid, "")
                        
                        candidates.append({
                            'id': match['id'],
                            'text': description,
                            'metadata': match['metadata'],
                            'original_score': match['score']
                        })
                    
                    # Re-rank
                    ranked_results = ranker.rank(query, candidates, top_k=12)
                    
                    st.markdown(f"Found **{len(ranked_results)}** matches for *'{query}'* (Re-ranked from top 50)")
                    
                    # Display results in a grid
                    cols = st.columns(3)
                    for idx, match in enumerate(ranked_results):
                        meta = match['metadata']
                        score = match['score']
                        
                        # Resolve image path
                        img_path = os.path.join(os.path.dirname(__file__), meta['path'])
                        
                        with cols[idx % 3]:
                            if os.path.exists(img_path):
                                image = Image.open(img_path)
                                st.image(image, width="stretch", caption=f"Score: {score:.2f}")
                            else:
                                st.warning(f"Image not found: {meta['path']}")
                else:
                    st.info("No matches found.")
            else:
                st.error("Failed to generate embedding for query.")

if __name__ == "__main__":
    main()
