import os
import pandas as pd
import streamlit as st
import txtai
import numpy

# Ensure duplicate library error is avoided
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Cache data loading
@st.cache_data
def load_data(chunk_size=10000):
    url = 'https://mystoragekedar.blob.core.windows.net/dataset/train.csv?sp=r&st=2025-03-07T04:19:57Z&se=2025-03-14T11:19:57Z&spr=https&sv=2022-11-02&sr=b&sig=19aHe5Apdm14IhER3FeAQcWB8lkP8j0Ipo2cqHXDYQc%3D'
    return url

# Initialize txtai embeddings
embeddings = txtai.Embeddings()

# Check if embeddings.db exists, if not, ask user to run embedding script
if os.path.exists("embeddings.db"):
    st.write("Loading existing embeddings...")
    embeddings.load("embeddings.db")
else:
    st.write("Error: Embeddings file not found! Run `embedding.py` first.")

# Streamlit UI
st.title('Amazon Item Search Engine')

query = st.text_input('Enter a Search Query:', '')

# Search function in chunks
def search_in_chunks(query, chunk_size=10000):
    url = load_data()
    
    search_results = []
    global_index = 0  # To keep track of the global index
    
    # Read the CSV file in chunks
    for chunk in pd.read_csv(url, chunksize=chunk_size):
        chunk = chunk.dropna()  # Drop rows with missing values
        
        # Perform search on current chunk
        results = embeddings.search(query, 5)
        
        for idx, score in results:
            # Adjust the idx to be local to the current chunk
            local_idx = idx - global_index
            
            # Check if the local_idx is within the bounds of the current chunk
            if local_idx >= 0 and local_idx < len(chunk):
                product_title = chunk.iloc[local_idx]['TITLE']
                search_results.append((product_title, score))
        
        # Update the global index for the next chunk
        global_index += len(chunk)
    
    return search_results

# Perform search when button is pressed
if st.button('Search'):
    if query:
        results = search_in_chunks(query)

        if results:
            st.write("### Search Results:")
            for i, (product_title, score) in enumerate(results):
                st.write(f"{i+1}. {product_title} (Score: {score:.4f})")
        else:
            st.write("No results found.")
    else:
        st.write('Please enter a query.')
