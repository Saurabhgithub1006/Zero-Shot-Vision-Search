import streamlit as st
import os
import sys
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.search import build_search_service

# Page Config
st.set_page_config(
    page_title="Driving Scenario Search",
    page_icon="🚗",
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
def load_service():
    return build_search_service()


EXAMPLES = {
    "a2d2": "e.g., 'a cyclist next to parked cars' or 'a truck on a country road'",
    "unsplash": "e.g., 'a futuristic city at night' or 'a happy dog running'",
}

ATTRIBUTION = {
    "a2d2": "Images: A2D2 – Audi Autonomous Driving Dataset, © Audi AG, "
            "licensed under [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/). "
            "Images are shown unmodified.",
    "unsplash": "Images: Unsplash Lite dataset.",
}


def main():
    st.title("🚗 Driving Scenario Search")
    st.markdown("### Zero-Shot Semantic Search over German Road Scenes")

    try:
        service = load_service()
    except Exception as e:
        st.error(f"Error loading components: {e}")
        st.stop()

    dataset = service.dataset
    query = st.text_input("Describe the driving scenario you're looking for...",
                          placeholder=EXAMPLES.get(dataset.name, ""))

    # Ground-truth classes from the label masks can narrow the search
    selected_labels = []
    if dataset.has_labels:
        all_labels = sorted({l for r in service.records.values() for l in r.labels})
        selected_labels = st.multiselect("Must contain (optional)", all_labels)

    if query:
        with st.spinner("Searching..."):
            try:
                results = service.search(query, top_k=12, labels=selected_labels)
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.stop()

        if not results:
            st.info("No matches found.")
        else:
            note = f" (re-ranked from top {service.candidate_k})" if service.ranker else ""
            st.markdown(f"Found **{len(results)}** matches for *'{query}'*{note}")

            cols = st.columns(3)
            for idx, result in enumerate(results):
                img_path = os.path.join(os.path.dirname(__file__), result.path)
                with cols[idx % 3]:
                    image = Image.open(img_path) if os.path.exists(img_path) else result.source_url
                    if image:
                        st.image(image, width="stretch", caption=f"Score: {result.score:.3f}")
                        if result.labels:
                            st.caption(", ".join(result.labels))
                    else:
                        st.warning(f"Image not found: {result.path}")

    st.markdown("---")
    st.caption(ATTRIBUTION.get(dataset.name, ""))

if __name__ == "__main__":
    main()
