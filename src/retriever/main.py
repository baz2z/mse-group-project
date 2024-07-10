import sys

from pathlib import Path
sys.path.insert(0, Path(__file__).resolve().parents[1])
print(Path(__file__).resolve().parents[1])

# internal imports:
from text_embedding import TextEmbedding
from index import Index
from bm25 import BM25
from fast_bm25 import FastBM25
from colBERT import colBERT
from in_out import load_corpus_from_json_files, load_url_mapping_from_csv, load_url_from_json_files
import time

def dummy_usage():
    
    # Example usage
    corpus = {
        'doc1': 'This is a sample document of french bulldogs.',
        'doc2': 'Another french sample document.',
        'doc3': 'This french document contains sample words.',
        'doc4': 'That document contains sample words again.',
        'doc5': 'This doc consists of example words about french bulldogs.',
        'doc6': 'This is a test document. Bulldogs are really cute.',
        'doc7': 'This is a french testing document for french classes',
        'doc8': 'Bulldogs are really cool. Except the french ones. They are too french.',
    }
    index_name = 'test_index'
    index = Index(corpus)
    index.initialize_index()
    index.export_index(index_name)
    
    # tfidf ranker class using sklearn fit transform
    index_path = Path("dat", f"{index_name}.json")
    bm25 = BM25(index_path)
    
    ranked_docs = bm25.rank('french bulldog')
    print(ranked_docs)
    

def create_index(corpus, index_name, num_splits, exist_ok=True):
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
    path_to_index: Path = Path("dat", f"{index_name}.json")
    if exist_ok and path_to_index.exists():
        print(f"Index file '{index_name}.json' found. Using pre-computed index")
        return path_to_index
    
    index = Index(corpus)
    index.initialize_index()
    index.export_index_numpy(index_name, num_splits)
    
    return path_to_index
    

def main():    
    
    # Try on crawled tuebingen data
    directory_path = "dat/crawler_test_run/crawler/index/"
    corpus_tue = load_corpus_from_json_files(directory_path)
    
    mapping_path = "dat/crawler_test_run/crawler/20240703010421.csv"
    url_mapping = load_url_mapping_from_csv(mapping_path)
    
    # ! Set exist_ok to False if corpus has changed
    index_name = "index_tue_v3"
    path_to_index = create_index(corpus_tue, index_name, exist_ok=False)
    
    query = 'Wo kann man gut Wein trinken in Tübingen?'
    colBERT_ranker = colBERT(path_to_index)
    # bm25 = BM25(path_to_index)
    ranked_docs = colBERT_ranker.rank(query=query)
    
    top5 = ranked_docs[:5]
    print(f"Top 5 documents for query: {query}")
    for doc_id, score in top5:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping.get(doc_id)}")


def main_extended_index():
    
    k = 15000
    directory_path = "dat/index/"
    corpus_tue = load_corpus_from_json_files(directory_path, k)
    url_mapping = load_url_from_json_files(directory_path, k)
    
    num_splits = k // 1000
    index_name = f"index_tue_extended_{k}"
    index_path = create_index(corpus_tue, index_name, num_splits, exist_ok=True)
    
    query = 'brecht hölderlin'
    # colBERT_ranker = colBERT(path_to_index)
    bm25 = BM25(index_path=index_path, index_name=index_name)
    ranked_docs = bm25.rank(query=query)
    
    top5 = ranked_docs[:5]
    print(f"Top 5 documents for query: {query}")
    for doc_id, score in top5:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping.get(doc_id)}")


def main_fast_bm25():
    
    k = 15000
    directory_path = "dat/index/"
    corpus_tue = load_corpus_from_json_files(directory_path, k)
    url_mapping = load_url_from_json_files(directory_path, k)
    
    emb = TextEmbedding(corpus=corpus_tue)
    doc_ids = list(emb.get_doc_ids())
    corpus = emb.get_corpus() 
    #TODO: reduce time to preprocess corpus (split etc.) 
       
    #TODO: check if already exists
    path = f"dat/fast_bm25_{k}"
    fastBM25 = FastBM25(corpus)
    fastBM25.save(path)
    
    query = 'brecht hölderlin'
    query = emb.bag_of_words(query).split(" ")

    start_time = time.time()

    fastBM25_load = FastBM25.load("dat/fast_bm25")
    top_n = fastBM25_load.get_top_n(query, doc_ids, n=5)

    end_time = time.time()

    for doc_id, score in top_n:
        print(f"Document ID: {doc_id}, Score: {score:.4f}, URL: {url_mapping.get(doc_id)}")

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    # dummy_usage()
    # main()
    # main_extended_index()
    main_fast_bm25()