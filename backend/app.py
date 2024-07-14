# app.py
import sys
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from retriever.main import mocked_retrieve

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
def get_search_results(query: str):
    results = mocked_retrieve(query)  # Assuming your main function takes a query as an argument
    # retrieve(corpus_path, corpus, url_mapping, query, reranker="NLI", 
    #          n_bm25_docs=100, bert_index_dir = "bert_embeddings"):
    print(f"results are: {results}")
    return {"results": results}
