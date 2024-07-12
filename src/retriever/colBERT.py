import numpy as np
import math
import json
import sys
import os
import pickle   
import torch

from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parents[1])

# internal imports
from text_embedding import BertEmbedding


class colBERT():
    """
    BM25 (Best Matching 25) is a ranking algorithm used for information retrieval that 
    computes relevance scores for documents given a query based on the tfidf approach.
    """
    
    def __init__(self, corpus):
        """
        Initialize BM25 on given pre computed index.
        
        Args:
            path (str): The file path to the index.   
        """
        self.ranker = 'colBERT'
        self.corpus = corpus
        self.text_embedding = BertEmbedding(corpus=corpus)
        self.bert_embeddings = None
        self.doc_ids = None
        # self.k = k
       
        # Initialize index
        # self.corpus_path = corpus_path        
        self.create_index()

    @staticmethod
    def load(filename):
        with open(f"{filename}.pkl", "rb") as fsave:
            return pickle.load(fsave)

    def save(self, filename):
        with open(f"{filename}.pkl", "wb") as fsave:
            pickle.dump(self, fsave, protocol=pickle.HIGHEST_PROTOCOL)
        
    def create_index(self):
        self.doc_ids, self.bert_embeddings =  self.text_embedding.get_bert_embeddings()

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
        scores, indices = self.compute_scores(query_vector, top_k)

        # Assuming self.doc_ids stores document IDs corresponding to the indices in self.bert_embeddings
        doc_ids = [self.doc_ids[idx] for idx in indices]
    
        ranked_docs = list(zip(doc_ids, scores))

        
        return ranked_docs


    def compute_scores(self, query_vector, top_k=10):
        sim_scores_per_doc = []
        epsilon = 1e-10  # Small value to avoid division by zero

        # print(self.bert_embeddings.shape)
        for query_embedding in query_vector:
            # print(query_embedding.shape)
            # Normalize query_embedding
            query_embedding_norm = query_embedding / (torch.norm(query_embedding, p=2, dim=0, keepdim=True) + epsilon)
            
            # Normalize bert_embeddings
            bert_embeddings_norm = self.bert_embeddings / (torch.norm(self.bert_embeddings, p=2, dim=2, keepdim=True) + epsilon)
            
            # Perform dot product using matmul, now with normalized embeddings
            scores = torch.matmul(bert_embeddings_norm, query_embedding_norm.T)  # Transpose query_embedding_norm for matmul
            # print(scores.shape)
            max_scores, _ = torch.max(scores, dim=1)
            sim_scores_per_doc.append(max_scores)
        
        # Convert list to tensor
        sim_scores_tensor = torch.stack(sim_scores_per_doc)
        # Sum over the documents
        sum_over_docs = torch.sum(sim_scores_tensor, dim=0)
        top_k_scores, top_k_indices = torch.topk(sum_over_docs, k=top_k)

        return top_k_scores, top_k_indices
    
    def get_top_k(self, scores, top_k):
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    