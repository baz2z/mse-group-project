import numpy as np
import math
import json
import sys

from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parents[1])

# internal imports
from text_embedding import TextEmbedding
from in_out import load_csr_matrix


class colBERT():
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
        self.ranker = 'colBERT'
        self.text_embedding = TextEmbedding()
       
        # Initialize index
        self.index_path = index_path        
        self.initialize_index()
        

    def initialize_index(self):
        """
        Initializes the index by loading the necessary data from the index json file.
        """
        with open(self.index_path, 'r') as file:
            index_data = json.load(file)
        
        self.doc_ids = index_data['doc_ids']
        self.bert_embedding = index_data['bert_embeddings']

    def vectorize_query(self, query):
        """
        Vectorizes the given query by creating a query vector based on the tokens in the query.

        Args:
            query (str): The query string.

        Returns:
            numpy.ndarray: The query vector representing the query.

        """
        return self.text_embedding.get_single_bert_embedding(query)
                
    
    def rank(self, query, top_k=10):
        """
        Rank the documents based on the given query.

        Args:
            query (str): The query string.
            top_k (int, optional): The number of documents to return. Defaults to 5.

        Returns:
            list: A list of tuples containing the document ID and the relevance score.
        """
        query_vector = self.vectorize_query(query)
        scores = self.compute_scores(query_vector)
        ranked_docs = self.get_top_k(scores, top_k)
        
        return ranked_docs
    
    def compute_scores(self, query_vector):
        scores = []
        for doc_id, doc_embedding in self.bert_embedding.items():
            doc_score = 0
            for query_token_embedding in query_vector[0]:
                token_similarities = []
                doc_token_embeddings = doc_embedding[0]
                # Calculate cosine similarities for this token across all document tokens
                similarities = [np.dot(query_token_embedding, doc_token_embedding) / 
                               (np.linalg.norm(query_token_embedding) * np.linalg.norm(doc_token_embedding)) 
                                for doc_token_embedding in doc_token_embeddings]  # Iterate through embeddings for each token
                token_similarities.append(max(similarities))  # Find the max similarity for this token
                doc_score += sum(token_similarities)  # Sum of max similarities for all query tokens
            scores.append((doc_id, doc_score))
        return scores
    
    def get_top_k(self, scores, top_k):
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    