"""FastAPI application for the backend API."""
import sys
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from retriever_v2 import main
from query_expansion import expand_query_with_wordnet
from preprocess import sanitize_html, extract_title, preprocess_dist_string, extract_description, extract_first_paragraph

app = FastAPI()
# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust according to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search")
def get_search_results(query: str, category: str = None):
    """
    Execute a search query and return the results.

    :params query: The query string to search for.
    :params category: The category of the query. If provided, the query will be expanded using WordNet.
    """
    print(f"Initial query: {query}")
    print(f"Category: {category}")
    if category:
        query = expand_query_with_wordnet(query, category)
        print(f"New Query to be sent to retrieval model: {query}")
    results = []
    
    ensemble_retriever = main.EnsembleRetriever(use_fast=False, pre_k=256)
    results = ensemble_retriever.query(query)
    
    # Calculate min and max scores
    min_score = results['score'].min()
    max_score = results['score'].max()

    results['dist'] = results['dist'].apply(lambda x: str(x.tolist()))
    
    # Prepare the response
    response = {
        'top_results': results.to_dict(orient='records'),
        'min_score': min_score,
        'max_score': max_score
    }
    
    return response
