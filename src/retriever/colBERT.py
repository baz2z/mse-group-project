import numpy as np
import math
import json
import sys
import os

from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parents[1])

# internal imports
from text_embedding import BertEmbedding


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
        self.text_embedding = BertEmbedding()
       
        # Initialize index
        self.index_path = index_path        
        self.initialize_index()

    @staticmethod
    def load(filename):
        with open(f"{filename}.pkl", "rb") as fsave:
            return pickle.load(fsave)
            
    def save(self, filename):
        with open(f"{filename}.pkl", "wb") as fsave:
            pickle.dump(self, fsave, protocol=pickle.HIGHEST_PROTOCOL)
        

    def initialize_index(self):
        """
        Initializes the index by loading the necessary data from the data/index_bert folder.
        Each subfolder represents a document ID, and each contains a 'bert_embeddings.npy' file.
        """
        self.doc_ids = []
        self.bert_embeddings = []

        # Path to the folder containing the index data
        index_folder_path = self.index_path
        
        # List all directories in the index folder
        for doc_id in os.listdir(index_folder_path):
            doc_path = os.path.join(index_folder_path, doc_id)
            
            # Check if the path is indeed a directory
            if os.path.isdir(doc_path):
                self.doc_ids.append(doc_id)
                
                # Path to the numpy array file
                embeddings_file_path = os.path.join(doc_path, 'bert_embedding.npy')
                print(f"Searching for file at: {os.path.abspath(embeddings_file_path)}")

                
                # Load the numpy array and append it to the bert_embeddings list
                self.bert_embeddings.append(np.load(embeddings_file_path))

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
        for doc_id, doc_embedding in zip(self.doc_ids, self.bert_embeddings):
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
    