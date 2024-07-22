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

    # result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'example_food_and_drinks.csv'))
    # results = pd.read_csv(result_path)
    # # Apply the preprocessing function to the 'dist' column
    # results['dist'] = results['dist'].apply(preprocess_dist_string)
    ensemble_retriever = main.EnsembleRetriever(use_fast=True)
    results = ensemble_retriever.query(query)

    # Calculate min and max scores
    min_score = results['score'].min()
    max_score = results['score'].max()

    # Prepare the response
    response = {
        'top_results': [],
        # 'html': [],
        'min_score': min_score,
        'max_score': max_score
    }

    html_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'retriever_v2', 'index', 'html'))
    
    for index, row in tqdm(results.iterrows(), total=results.shape[0], desc="Processing HTML files"):
        doc_id = row['doc_id']
        html_path = os.path.join(html_dir, f"{doc_id}.html")
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            title = extract_title(html_content)
            description = extract_description(html_content) or extract_first_paragraph(html_content)
            # sanitized_html = sanitize_html(html_content)
            
            row_dict = row.to_dict()
            row_dict['title'] = title
            row_dict['description'] = description
            response['top_results'].append(row_dict)
            # response['html'].append(sanitized_html)
    
    return response
