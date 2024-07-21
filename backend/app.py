# app.py
import sys
import os
from pathlib import Path
import time 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# from retriever import main
from retriever_v2 import main
import pandas as pd

app = FastAPI()
# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust according to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_dist_string(dist_string):
    # Remove leading and trailing spaces and brackets
    dist_string = dist_string.strip().strip('[]')
    
    # Replace multiple spaces with a single comma
    dist_string = ','.join(part.strip() for part in dist_string.split())
    
    # Add brackets around the string to ensure it's a valid JSON array format
    return f'[{dist_string}]'


@app.get("/search")
def get_search_results(query: str):
    results = []

    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'example_food_and_drinks.csv'))
    results = pd.read_csv(result_path)
    # Apply the preprocessing function to the 'dist' column
    results['dist'] = results['dist'].apply(preprocess_dist_string)

    print(f"results.iloc[:10][dist]: {results.iloc[:10]['dist']}")

    # Calculate min and max scores
    min_score = results['score'].min()
    max_score = results['score'].max()

    # Convert DataFrame to a list of dictionaries
    results_list = results.to_dict(orient='records')

    # Prepare the response
    response = {
        'top_results': results_list,
        'min_score': min_score,
        'max_score': max_score
    }

    # results =  [
    #   {
    #     'url': 'https://scholar.google.com/citations?user=QQi1_rAAAAAJ&hl=ja',
    #     'score': 0.94,
    #     'created:': '2024-07-14 18:45:53.202173',
    #     'title': 'Example Page 1',
    #     'abstract': 'This is a summary of example page 1.',
    #   },
    #   {
    #     'url': 'https://www.tuebingen-info.de',
    #     'score': 0.89,
    #     'created:': '2024-07-14 18:45:53.202173',
    #     'title': 'Example Page 2',
    #     'abstract': 'This is a summary of example page 2.',
    #   },
    # {
    #     'url': 'https://neckarmueller.de/',
    #     'score': 0.5,
    #     'created:': '2024-07-14 18:45:53.202173',
    #     'title': 'Example Page 3',
    #     'abstract': 'This is a summary of example page 3.',
    #   },
    # ]
    
    return response
