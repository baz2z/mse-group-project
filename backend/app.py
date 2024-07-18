# app.py
import sys
import os
from pathlib import Path
import time 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from retriever import main

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
    results = []
    # results = main.mocked_retrieve(query)  # Assuming your main function takes a query as an argument
    
    start_time = time.time()
    
    retrieval_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) / "retriever/dat/"
    index_dir = retrieval_path / "index/"
    index_mapping = retrieval_path / "index_mapping.csv"
    k_docs = 20993
    # print(f"index_dir {index_dir}")
    # print(f"index_mapping {index_mapping}")
    corpus_tue =  main.load_corpus(index_dir, k_docs)
    url_mapping = main.load_url_mapping_from_csv(index_mapping)
    print(f"len(url_mapping): {len(url_mapping)}")

    top_n, url_mapping_top_n = main.retrieve(corpus_path=index_dir, corpus=corpus_tue, url_mapping=url_mapping, query=query, reranker="NLI", n_bm25_docs=100)

    for id_num, (doc_id, score) in enumerate(top_n):
        print(f"id_num: {id_num}, url_mapping_top_n.get(doc_id): {url_mapping_top_n.get(doc_id)}, score: {score:.4f}")
        results.append({"url": url_mapping_top_n.get(doc_id), "score": score})
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return results
