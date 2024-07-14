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
from retriever.text_embedding import BagOfWordsTokenizer, BertEmbedding
from retriever.bm25 import BM25
from retriever.colBERT import colBERT
from retriever.nli import DebertaV3
from retriever.in_out import load_corpus_from_json_files, load_url_from_json_files, load_corpus, load_url_mapping_from_csv


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
        print(f"\nBM25 file '{index_name}_{k}.pkl' found. Using pre-computed BM25")
        bm25 = BM25.load(path)
        return bm25
    
    tokenized_corpus = embed.tokenize_corpus() 
    bm25 = BM25(tokenized_corpus)
    bm25.save(path)
    
    return bm25


def create_colBERT(corpus_path, index_path, doc_ids=[], exist_ok=True):

    path = f"dat/{index_path}"
    embeddings_exist = Path(f"{path}/").exists()
    
    colbert = colBERT(corpus_path, path, doc_ids)
    
    if exist_ok and embeddings_exist:
        print(f"\ncolBERT embeddings found at {path}. Using pre-computed colBERT")
        colbert.load()
        return colbert
    
    colbert.create()
    colbert.load()
    
    return colbert


def main_bert():
    start_time = time.time()
    k = 200 #  how many docs to filter through
    corpus_path = "dat/eng_subset/"  
    mapping_path = "dat/eng_doc_id_mapping.csv"
    url_mapping = load_url_mapping_from_csv(mapping_path)
    
    index_folder = "index_bert_docs"
    doc_ids = ['9b800db5aec8739c6e714182f06628fc', '689d133e430a98a71cdbedd49e3b45bc', '92076b1cf6097ae822d821e46fb84fe0', 
               '458ec03ec352ab4c0964b2416219c200', '1883368a03590035284d4b98590de999', '1d78b99d44cfbe741f47d35d943a138d', 
               '2ff8d17df1f12538c888788fa188253b', '3d737ae492c4c07ffa103c8ce12ba771', '310ad012b83cd9a71eb3c08dc8b5bdb6', 
               '34d3e4d5d2b95fe943d5595ab7ee3a84']
    
    query = 'Amazon vouchers'
    bert = create_colBERT(corpus_path, index_folder, k, doc_ids, exist_ok=True)
    ranked_docs = bert.rank(query=query, top_k=5)
    
    print(f"Top 5 documents for query: {query}")
    for doc_id, score in ranked_docs:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping.get(doc_id)}")    
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

def main_bm25():
    
    # Load k crawled documents 
    k = 20993
    directory_path = "dat/index/"
    mapping_path = "dat/index_mapping.csv"
    corpus = load_corpus(directory_path, k)
    url_mapping = load_url_mapping_from_csv(mapping_path)

    start_time = time.time()

    # Init tokenization util on given corpus
    embed = BagOfWordsTokenizer(corpus=corpus)
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
    

def retrieve(corpus_path, corpus, url_mapping, query, reranker="NLI", 
             n_bm25_docs=100, bert_index_dir = "bert_embeddings"):
    
    print("\n\nQuery: ", query)
    embed = BagOfWordsTokenizer(corpus=corpus)
    doc_ids = embed.doc_ids
    query_tokenized = embed.tokenize(query)
    
    bm25_name = "bm25_eng"
    bm25 = pre_compute_bm25(embed, bm25_name, len(doc_ids), exist_ok=True)

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
        print("\n\nReranking with DebertaV3 (NLI)")
        deberta = DebertaV3(corpus_top_n)
        top_n = deberta.rank(query, top_k=5)
    if reranker == "colBERT":
        print("\n\nReranking with colBERT")
        colbert_name = "colBERT_v4"
        
        colbert = create_colBERT(corpus_path, bert_index_dir, k, doc_ids, exist_ok=True)
        ranked_docs = colbert.rank(query=query, top_k=5)
    
        # colbert = create_colBERT(corpus_top_n, colbert_name, len(corpus_top_n), exist_ok=False)
        # top_n = colbert.rank(query, top_k=5)
        # colbert.bert_embeddings = None # free up memory
    
    print("\n\nFinal top documents for query: ", query)    
    for doc_id, score in top_n:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping_top_n.get(doc_id)}")
    
    return top_n, url_mapping_top_n

def mocked_retrieve(query):
    return [
      {
        'link': 'https://uni-tuebingen.de/',
        'title': 'Example Page 1',
        'abstract': 'This is a summary of example page 1.',
      },
      {
        'link': 'https://www.tuebingen-info.de',
        'title': 'Example Page 2',
        'abstract': 'This is a summary of example page 2.',
      },
    ]


def batch(
    corpus, 
    corpus_path,
    url_mapping, 
    batch_of_queries, 
    results_path,
    reranker="NLI",
    n_bm25_docs=100,
    ):
    synonyms_tue = ["tuebingen", "tübingen", "Tuebingen", "Tübingen"]
    
    # Load queries from the query batch file
    with open(batch_of_queries, 'r') as file:
        queries = file.readlines()
        queries = [query.strip() for query in queries]
        # add "tuebingen" to each query if not already present else original query
        queries = [
            query + " tuebingen" 
            if not any(synonym in query for synonym in synonyms_tue) 
            else query for query in queries
            ]
        
    with open(results_path, 'w') as results_file:
            results_file.write("query\tdocument\turl\tscore\n")
        
    for query_num, query in enumerate(queries):
        with open(results_path, 'a') as results_file:
        
            # Get the top n documents for each query
            top_n, url_mapping_top_n = retrieve(
                corpus=corpus,
                corpus_path=corpus_path,
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
    # main_bert()
    # main_bm25()
    # main_nli()
    
    start_time = time.time()
    
    index_dir = "dat/index/"
    index_mapping = "dat/index_mapping.csv"
    k_docs = 20993
    
    # Load k crawled documents 
    corpus_tue = load_corpus(index_dir, k_docs)
    url_mapping = load_url_mapping_from_csv(index_mapping)
    
    batch(
        corpus=corpus_tue,
        corpus_path=index_dir,
        url_mapping=url_mapping,
        batch_of_queries="dat/query_batch_file.txt",
        results_path="dat/results.txt",
        reranker="NLI", # one of "colBERT" or "NLI"
        n_bm25_docs=100,
        )
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    