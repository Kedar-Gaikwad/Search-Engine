import os
import pandas as pd
import streamlit as st
import txtai
import numpy

# Ensure duplicate library error is avoided
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Cache data loading
@st.cache_data
def load_data():
    path = r'D:\data and text mining\AS4\archive\dataset\train.csv'
    df = pd.read_csv(path)
    return df.dropna()

# Load dataset
titles = load_data()

# Initialize txtai embeddings
embeddings = txtai.Embeddings()
if os.path.exists("embeddings.db"):
    st.write("Loading existing embeddings...")
    embeddings.load("embeddings.db")
else:
    st.write("Error: Embeddings file not found! Run `embedding.py` first.")

# Streamlit UI
st.title('Amazon Item Search Engine')

query = st.text_input('Enter a Search Query:', '')

if st.button('Search'):
    if query:
        results = embeddings.search(query, 5)

        if results:
            st.write("### Search Results:")
            for i, (idx, score) in enumerate(results):
                st.write(f"{i+1}. {titles.iloc[idx]['TITLE']} (Score: {score:.4f})")
        else:
            st.write("No results found.")
    else:
        st.write('Please enter a query.')
