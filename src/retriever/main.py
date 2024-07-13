import sys
import time
import os
import json
import gc

from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0, Path(__file__).resolve().parents[1])
print(Path(__file__).resolve().parents[1])

# internal imports:
from text_embedding import BagOfWordsTokenizer, BertEmbedding
from bm25 import BM25
from colBERT import colBERT
from nli import DebertaV3
from in_out import load_corpus_from_json_files, load_url_from_json_files


def pre_compute_bm25(embed, index_name, k, exist_ok=True):
    """
    Pre-computes and saves the BM25 object for a given embedding, index name, and k value.
    
    Args:
        embed (Embedding): The embedding object used for tokenization.
        index_name (str): The name of the index.
        k (int): The k value used for BM25 computation.
        exist_ok (bool, optional): If True, allows reusing pre-computed BM25 if it exists. 
            Defaults to True.
    
    Returns:
        BM25: The pre-computed BM25 object.
    """
    path = f"dat/{index_name}_{k}"
    bm25_exists = Path(f"{path}.pkl").exists()
    if exist_ok and bm25_exists:
        print(f"\n\nBM25 file '{index_name}_{k}.pkl' found. Using pre-computed BM25")
        bm25 = BM25.load(path)
        return bm25
    
    tokenized_corpus = embed.tokenize_corpus() 
    bm25 = BM25(tokenized_corpus)
    bm25.save(path)
    
    return bm25


def create_colBERT(corpus_path, index_path, k, doc_ids=[], exist_ok=True):

    path = f"dat/{index_path}"
    embeddings_exist = Path(f"{path}/").exists()
    colbert = colBERT(corpus_path, path, doc_ids)
    if exist_ok and embeddings_exist:
        print(f"colBERT file '{path}' found. Using pre-computed colBERT")
        colbert.load()
        return colbert
    
    colbert.create()
    colbert.load()
    
    return colbert


@contextmanager
def managed_colbert(*args, **kwargs):
    colbert = create_colBERT(*args, **kwargs)
    try:
        yield colbert
    finally:
        # Clear any internal collections or resources
        colbert.bert_embeddings = None  # free up memory
        del colbert
        gc.collect()
        
    
def get_doc_url(doc_id, directory_path):
    # Construct the full path to the JSON file
    file_path = os.path.join(directory_path, f"{doc_id}.json")
    
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Return the value of the "url" key
    return data["url"]


def main_bert():
    start_time = time.time()
    k = 10 #  how many docs to filter through
    corpus_path = "dat/crawled_test2/"  
    index_path = "index_bert_docs"

    doc_ids = [ "0a1a5f1a7f5081adcb07c1f97ef77913",  "0a1b6985dd586ec83d474a6371dc926e",  
                "0a1e9d2ad6ff9b371b6955bcd19f96ae",  "0a1f82668cb5b821a5eb16bbf0270563",
                "0a2baaaadff9030f3d5d6c858bd55124"]

    bert = create_colBERT(corpus_path, index_path, k, doc_ids, exist_ok=False)
    query = 'this text is about food'
    ranked_docs = bert.rank(query=query, top_k=5)
    
    print(f"Top 5 documents for query: {query}")
    for doc_id, score in ranked_docs:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {get_doc_url(doc_id, corpus_path)}")    
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

def main_bm25():
    
    # Load k crawled documents 
    k = 15000
    directory_path = "dat/crawled_docs/"
    # TODO: Find more efficient way to load the corpus and url mapping
    corpus_tue = load_corpus_from_json_files(directory_path, k)
    url_mapping = load_url_from_json_files(directory_path, k)

    start_time = time.time()

    # Init tokenization util on given corpus
    embed = BagOfWordsTokenizer(corpus=corpus_tue)
    doc_ids = embed.doc_ids
    
    # Load or create precomputed BM25 scores (tf and idf scores)
    bm25_name = "bm25"
    bm25 = pre_compute_bm25(embed, bm25_name, k, exist_ok=False)
    
    # Query tokenization
    query = 'hölderlin'
    query = embed.tokenize(query)

    # Retrieve top n documents for the given query
    top_n = bm25.retrieve_top_n(query, doc_ids, n=5)
    for doc_id, score in top_n:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping.get(doc_id)}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

def main_nli():
    k = 2000
    directory_path = "dat/crawled_docs/"
    corpus_tue = load_corpus_from_json_files(directory_path, k)
    url_mapping = load_url_from_json_files(directory_path, k)
    
    # Query
    query = 'hölderlin'
    
    start_time = time.time()
    
    # Init DebertaV3 model on given corpus
    deberta = DebertaV3(corpus_tue)
    top_n = deberta.rank(query, top_k=5)
    for doc_id, score in top_n:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping.get(doc_id)}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    

def retrieve(corpus, url_mapping, query, reranker="DebertaV3", n_bm25_docs=100):
    
    embed = BagOfWordsTokenizer(corpus=corpus)
    doc_ids = embed.doc_ids
    query_tokenized = embed.tokenize(query)
    
    bm25_name = "bm25"
    bm25 = pre_compute_bm25(embed, bm25_name, len(doc_ids), exist_ok=False)

    # Retrieve top n documents for the given query
    bm25_top_n = bm25.retrieve_top_n(query_tokenized, doc_ids, n=n_bm25_docs)
    print("Pre-ranking with BM25")
    for i, (doc_id, score) in enumerate(bm25_top_n):
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping.get(doc_id)}")
        if i == 5:
            break
        
    bm25_top_n_doc_ids = [doc_id for doc_id, _ in bm25_top_n]
    corpus_top_n = {doc_id: corpus[doc_id] for doc_id in bm25_top_n_doc_ids}
    url_mapping_top_n = {doc_id: url_mapping[doc_id] for doc_id in bm25_top_n_doc_ids}

    if reranker == "NLI":
        print("\n\nReranking with DebertaV3")
        deberta = DebertaV3(corpus_top_n)
        top_n = deberta.rank(query, top_k=5)
    if reranker == "colBERT":
        print("\n\nReranking with colBERT")
        colbert_name = "colBERT_v4"
        with managed_colbert(
            corpus_top_n, colbert_name, len(corpus_top_n), exist_ok=False
            ) as colbert:
            top_n = colbert.rank(query, top_k=5)
        # colbert = create_colBERT(corpus_top_n, colbert_name, len(corpus_top_n), exist_ok=False)
        # top_n = colbert.rank(query, top_k=5)
        # colbert.bert_embeddings = None # free up memory
    
    print("\n\nFinal top documents for query: ", query)    
    for doc_id, score in top_n:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping_top_n.get(doc_id)}")
    
    return top_n, url_mapping_top_n


def batch(
    corpus, 
    url_mapping, 
    batch_of_queries, 
    results_path,
    reranker="DebertaV3",
    n_bm25_docs=100,
    ):
    # Load queries from the query batch file
    with open(batch_of_queries, 'r') as file:
        queries = file.readlines()
        queries = [query.strip() for query in queries]
        # add "tuebingen" to each query
        queries = [query + " tuebingen" for query in queries]
    
    with open(results_path, 'w') as results_file:
            results_file.write("query\tdocument\turl\tscore\n")
        
    for query_num, query in enumerate(queries):
        with open(results_path, 'a') as results_file:
        
            # Get the top n documents for each query
            top_n, url_mapping_top_n = retrieve(
                corpus=corpus,
                url_mapping=url_mapping,
                query=query,
                reranker=reranker,
                n_bm25_docs=n_bm25_docs,
            )
            
            # Write the results to the results file
            for id_num, (doc_id, score) in enumerate(top_n):
                results_file.write(
                    f"{query_num}\t{id_num}\t{url_mapping_top_n.get(doc_id)}\t{score:.4f}\n"
                )
        
        
if __name__ == "__main__":
    main_bert()
    # main_bm25()
    # main_nli()
    start_time = time.time()
    
    index_dir = "dat/crawled_docs/"
    k_docs = 100
    
    # Load k crawled documents 
    corpus_tue = load_corpus_from_json_files(index_dir, k_docs)
    url_mapping = load_url_from_json_files(index_dir, k_docs)
    
    # batch(
    #     corpus=corpus_tue,
    #     url_mapping=url_mapping,
    #     batch_of_queries="dat/query_batch_file.txt",
    #     results_path="dat/results.txt",
    #     reranker="colBERT", # one of "colBERT" or "NLI"
    #     n_bm25_docs=10,
    #     )
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    