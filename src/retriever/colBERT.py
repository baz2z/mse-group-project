import numpy as np
import math
import json
import sys
import os
import pickle   
import torch
import gzip

from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parents[1])

# internal imports
from text_embedding import BertEmbedding


def save_tensor(tensor, file_path, compress=False):
	if compress:
		with gzip.open(file_path + '.gz', 'wb') as f:
			torch.save(tensor, f)
	else:
		torch.save(tensor, file_path)

def load_tensor(file_path, compressed=False):
	if compressed:
		with gzip.open(file_path, 'rb') as f:
			tensor = torch.load(f)
	else:
		tensor = torch.load(file_path)
	return tensor


class colBERT():
    """
    BM25 (Best Matching 25) is a ranking algorithm used for information retrieval that 
    computes relevance scores for documents given a query based on the tfidf approach.
    """
    
    def __init__(self, corpus_path, index_path, doc_ids):
        """
        Initialize BM25 on given pre computed index.
        
        Args:
            path (str): The file path to the index.   
        """
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.doc_ids = doc_ids
        self.bert_embeddings = None

    # TODO: Should not be static anymore since we are not laoding the entire class anymore
    @staticmethod
    def load(filename): # TODO: new param: doc_ids (based on bm25 output)
        
        # TODO:
        # Restack single doc tensors into batch tensor
        #  - load all tensors from pt files that match the provided doc_ids
        #  - stack them into a single tensor
        
        with open(f"{filename}.pkl", "rb") as fsave:
            return pickle.load(fsave)

    def save(self, filename):
        with open(f"{filename}.pkl", "wb") as fsave:
            pickle.dump(self, fsave, protocol=pickle.HIGHEST_PROTOCOL)

    def create(self):
        # BertEmbedding nur als klassenvariable wenn embeddings erzeugt werden
        text_embedding = BertEmbedding(self.corpus_path, self.index_path)
        self.create_index(text_embedding)
        
    def create_index(self, text_embedding):
        
        # TODO:
        # Loop through corpus -> for (doc_id, doc_text) in self.corpus.items():
        #  - Maybe tokenize each document (might be relevant for .md files with extra symbols)
        #    -> remove symbols (wir müssen uns dann einfach mal die .md files anschauen)
        #  - Get BERT embeddings for each document
        #    -> apply text_embedding.get_bert_embeddings() 
        #    -> oder auch einfach nur text_embedding.get_single_bert_embedding(doc_text)?
        #  - save single embedding to pt file (see save_tensor function up top)
        #    -> name should be something like f"dat/{bert_index_name}/{doc_id}.pt"
        text_embedding.initiate_corpus()
        self.doc_ids, self.bert_embeddings =  text_embedding.get_bert_embeddings()

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
        
        # TODO: 
        # Implement batching functionality
        # - initialize a list to store the scores for each batch
        # - laod a given batch using the self.load function (not static anymore)
        # - perform compute_scores on the batch
        #   -> this should only return the scores and not do the top_k selection
        #   -> outsource the top_k selection to a separate function
        # - keep track of score for given doc_ids and append them to the list
        # - top k selection and return
        
        scores, indices = self.compute_scores(query_vector, top_k)

        # Assuming self.doc_ids stores document IDs corresponding to the indices in self.bert_embeddings
        doc_ids = [self.doc_ids[idx] for idx in indices]
    
        ranked_docs = list(zip(doc_ids, scores))

        
        return ranked_docs


    def compute_scores(self, query_vector, top_k=10):
        sim_scores_per_doc = []
        epsilon = 1e-10  # Small value to avoid division by zero

        print("Embedding",self.bert_embeddings.shape)
        for query_embedding in query_vector:
            print("Query Embed",query_embedding.shape)
            # Normalize query_embedding
            query_embedding_norm = query_embedding / (torch.norm(query_embedding, p=2, dim=0, keepdim=True) + epsilon)
            
            # Normalize bert_embeddings
            bert_embeddings_norm = self.bert_embeddings / (torch.norm(self.bert_embeddings, p=2, dim=2, keepdim=True) + epsilon)
            
            # Perform dot product using matmul, now with normalized embeddings
            scores = torch.matmul(bert_embeddings_norm, query_embedding_norm.T)  # Transpose query_embedding_norm for matmul
            print(scores.shape)
            max_scores, _ = torch.max(scores, dim=1)
            sim_scores_per_doc.append(max_scores)
        
        # Convert list to tensor
        sim_scores_tensor = torch.stack(sim_scores_per_doc)
        # Sum over the documents
        sum_over_docs = torch.sum(sim_scores_tensor, dim=0)
        # Ensure top_k does not exceed the number of documents
        adjusted_top_k = min(top_k, sum_over_docs.size(0))
        top_k_scores, top_k_indices = torch.topk(sum_over_docs, k=adjusted_top_k)

        return top_k_scores, top_k_indices
    
    def get_top_k(self, scores, top_k):
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    