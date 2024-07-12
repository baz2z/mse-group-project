import sys
import time
import os
import json

from pathlib import Path
sys.path.insert(0, Path(__file__).resolve().parents[1])
print(Path(__file__).resolve().parents[1])

# internal imports:
from text_embedding import BagOfWordsTokenizer, BertEmbedding
from index import Index
from bm25 import BM25
from colBERT import colBERT
from nli import DebertaV3
from in_out import load_corpus_from_json_files, load_url_from_json_files


def create_index(corpus, index_name, exist_ok=True):
    """
    Create an index for the given corpus and export it to a JSON file.

    Args:
        corpus (list): The corpus to create the index from.
        index_name (str): The name of the index.
        exist_ok (bool, optional): If True, return the path to the existing index file
            if it already exists. If False, overwrite the existing index file. 
            Defaults to True.

    Returns:
        Path: The path to the index file.

    """
    # TODO: adapt to new folder structure for bert embeddings
    path_to_index: Path = Path("dat", f"{index_name}")
    if exist_ok and path_to_index.exists():
        print(f"Index file '{index_name}' found. Using pre-computed index")
        return path_to_index
    
    index = Index(corpus)
    index.initialize_index()
    index.export_index(index_name)
    
    return path_to_index


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
        print(f"BM25 file '{index_name}_{k}.pkl' found. Using pre-computed BM25")
        bm25 = BM25.load(path)
        return bm25
    
    tokenized_corpus = embed.tokenize_corpus() 
    bm25 = BM25(tokenized_corpus)
    bm25.save(path)
    
    return bm25

def create_colBERT(corpus_path, index_name, k, exist_ok=True):

    path = f"dat/{index_name}_{k}"
    colbert_exists = Path(f"{path}.pkl").exists()
    if exist_ok and colbert_exists:
        print(f"colBERT file '{index_name}_{k}.pkl' found. Using pre-computed colBERT")
        colbert = colBERT.load(path)
        return colbert
    

    colbert = colBERT(corpus_path)
    colbert.save(path)
    
    return colbert
    
def get_doc_url(doc_id, directory_path):
    # Construct the full path to the JSON file
    file_path = os.path.join(directory_path, f"{doc_id}.json")
    
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Return the value of the "url" key
    return data["url"]


def main_bert():   
    
     # Load k crawled documents 
    k = 5
    directory_path = "dat/crawled_test2/"
    corpus_tue = load_corpus_from_json_files(directory_path, k)
    # url_mapping = load_url_from_json_files(directory_path, k)
    index_name = "test2_index_bert"
    path_to_index = create_index(corpus_tue, index_name, exist_ok=True)

    query = 'geigerle'
    colBERT_ranker = colBERT(path_to_index)
    ranked_docs = colBERT_ranker.rank(query=query, top_k=5)
    
    print(f"Top 5 documents for query: {query}")
    for doc_id, score in ranked_docs:

        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {get_doc_url(doc_id, directory_path)}")


def main_bert_refactor():
    start_time = time.time()
    k = 10
    corpus_path = "dat/crawled_docs/"
    colBERT_name = "colBERT_v2"
    bert = create_colBERT(corpus_path, colBERT_name, k, exist_ok=False)
    query = 'this text is about culture'
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
    bm25 = pre_compute_bm25(embed, bm25_name, k, exist_ok=True)
    
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
    

def retrieve(index_dir, k_docs, query):
    
    start_time = time.time()
    
    # Load k crawled documents 
    corpus_tue = load_corpus_from_json_files(index_dir, k_docs)
    url_mapping = load_url_from_json_files(index_dir, k_docs)
    
    embed = BagOfWordsTokenizer(corpus=corpus_tue)
    doc_ids = embed.doc_ids
    query_tokenized = embed.tokenize(query)
    
    bm25_name = "bm25"
    bm25 = pre_compute_bm25(embed, bm25_name, k_docs, exist_ok=True)

    # Retrieve top n documents for the given query
    bm25_top_n = bm25.retrieve_top_n(query_tokenized, doc_ids, n=100)
    print("Pre-ranking with BM25")
    for i, (doc_id, score) in enumerate(bm25_top_n):
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping.get(doc_id)}")
        if i == 5:
            break
        
    bm25_top_n_doc_ids = [doc_id for doc_id, _ in bm25_top_n]
    corpus_top_n = {doc_id: corpus_tue[doc_id] for doc_id in bm25_top_n_doc_ids}
    url_mapping_top_n = {doc_id: url_mapping[doc_id] for doc_id in bm25_top_n_doc_ids}

    deberta = DebertaV3(corpus_top_n)
    deberta_top_n = deberta.rank(query, top_k=5)
    print("Reranking with DebertaV3")
    for doc_id, score in deberta_top_n:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping_top_n.get(doc_id)}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    return deberta_top_n, url_mapping_top_n


def batch(batch_of_queries, results_path):
    # Load queries from the query batch file
    with open(batch_of_queries, 'r') as file:
        queries = file.readlines()
        queries = [query.strip() for query in queries]
    
    with open(results_path, 'w') as results_file:
        results_file.write("query\tdocument\turl\tscore\n")
        
        # Get the top n documents for each query
        for query_num, query in enumerate(queries):
            top_n, url_mapping_top_n = retrieve(
                index_dir="dat/crawled_docs/",
                k_docs=15000,
                query=query
            )
            
            # Write the results to the results file
            for id_num, (doc_id, score) in enumerate(top_n):
                results_file.write(
                    f"{query_num}\t{id_num}\t{url_mapping_top_n.get(doc_id)}\t{score:.4f}\n"
                )
        
        
if __name__ == "__main__":
    # main_bert()
    # main_bm25()
    # main_nli()
    # retrieve(
    #     index_dir="dat/crawled_docs/", 
    #     k_docs=15000, 
    #     query="cinema"
    #     )
    
    batch(
        batch_of_queries="dat/query_batch_file.txt",
        results_path="dat/results.txt"
        )