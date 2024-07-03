import numpy as np
import math
import json
import sys

from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parents[1])

# internal imports
from text_embedding import TextEmbedding
from in_out import load_csr_matrix


class BM25():
    """
    BM25 (Best Matching 25) is a ranking algorithm used for information retrieval that 
    computes relevance scores for documents given a query based on the tfidf approach.
    """
    
    def __init__(self, index_path):
        """
        Initialize BM25 on given pre computed index.
        
        Args:
            path (str): The file path to the index.   
        """
        self.ranker = 'bm25'
        self.text_embedding = TextEmbedding()
       
        # Initialize index
        self.index_path = index_path        
        self.initialize_index()
        
        # Initialize BM25 parameters
        self.N = len(self.doc_ids)
        self.avg_doc_len = sum(self.doc_lengths) / self.N
        
        self.k = 1.5  # default k in range [1.2, 2.0]
        self.b = 0.75 # default value for b

    def initialize_index(self):
        """
        Initializes the index by loading the necessary data from the index json file.
        """
        with open(self.index_path, 'r') as file:
            index_data = json.load(file)
        
        self.doc_ids = index_data['doc_ids']
        self.doc_lengths = index_data['doc_lengths']
        self.token_names = index_data['token_names']
        self.tf = load_csr_matrix(index_data['tfs'])
        self.idf = index_data['idfs']
        self.tfidf = load_csr_matrix(index_data['tfidfs'])

    def vectorize_query(self, query):
        """
        Vectorizes the given query by creating a query vector based on the tokens in the query.

        Args:
            query (str): The query string.

        Returns:
            numpy.ndarray: The query vector representing the query.

        """
        query_tokens = self.text_embedding.bag_of_words(query).split(' ')
        token_index = {token: i for i, token in enumerate(self.token_names)}
        
        # create a query vector out of query tokens
        query_vec = np.zeros(len(self.token_names))
        for token in query_tokens:
            index = token_index.get(token)
            if index is not None:
                query_vec[index] = 1
                
        return query_vec
    
    def rank_tfidf(self, query):
        """
        Ranks documents using TF-IDF scores.
        
        Args:
            query (str): The query string for which to calculate the TF-IDF scores.

        Returns:
            list: A list of tuples, where each tuple contains a document ID and its relevance score.
        """
        query_vector = self.vectorize_query(query)
        scores = self.tfidf.dot(query_vector)
        
        ranked_documents = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        
        return [
            (self.doc_ids[doc_id], score) 
            for doc_id, score in ranked_documents
        ]
    
    # TODO: vectorize for better performance
    def rank(self, query):
        """
        BM25 ranking algorithm. 
        
        Uses the tfidf approach to compute relevance scores for documents given a query. 
        
        Args:
            query (str): The query string.
            
        Returns:
            list: A list of tuples, where each tuple contains a document ID and its relevance score.
        """
        query_vector = self.vectorize_query(query)
        scores = np.zeros(len(self.doc_ids))
        
        for i, doc_id in enumerate(self.doc_ids):
            doc_length = self.doc_lengths[i]
            for j, term in enumerate(self.token_names):
                if query_vector[j] > 0:
                    idf = self.idf[term]
                    tf_qi_D = self.tf[i, j]
                    term_score = idf * (tf_qi_D * (self.k+1)) / \
                        (tf_qi_D + self.k * (1 - self.b + self.b * (doc_length / self.avg_doc_len)))
                    scores[i] += term_score
                    
        ranked_documents = sorted(
            [(self.doc_ids[idx], score) for idx, score in enumerate(scores)],
            key=lambda x: x[1], reverse=True
        )

        return ranked_documents
    