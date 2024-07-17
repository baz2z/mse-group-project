import numpy as np
import math
import json
import sys
import os
import pickle   
import torch
import gzip
import random   
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)

from sentence_transformers import SentenceTransformer
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

sys.path.insert(0, Path(__file__).resolve().parents[1])



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


class sentEmb():
    """
    BM25 (Best Matching 25) is a ranking algorithm used for information retrieval that 
    computes relevance scores for documents given a query based on the tfidf approach.
    """
    
    def __init__(self, corpus, index_path):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index_path = index_path
        self.corpus = corpus

    def create(self):
        for doc_id, document in self.corpus.items():
            # Split the document text by double new lines        
            sections = document.split("\n\n")

            sampled_sections = random.sample(sections, min(len(sections), 32))
            embeddings = []  # Initialize an empty list to collect embeddings

            for section in  sampled_sections:
                embedding = self.model.encode(section, convert_to_tensor=True)
                embeddings.append(embedding)



            embedding_size = len(embeddings[0])
            
            # If the number of sections is less than 32, pad the embeddings
            if len(sampled_sections) < 32:
                num_padding = 32 - len(sampled_sections)
                for _ in range(num_padding):
                    padding = torch.zeros(1, embedding_size)  # Create a single padding embedding
                    embeddings.append(padding.squeeze(0))  # Append each padding embedding individually
            
            document_embedding = torch.stack(embeddings)
            # print the document embedding size
            print(document_embedding.size())
            # Save the concatenated embeddings tensor
            file_path = f"{self.index_path}/{doc_id}.pt"
            torch.save(document_embedding, file_path)
        return


    def load(self):
        files = os.listdir(self.index_path)
        embeddings = []
        doc_ids = []

        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(self.index_path, file)
                embedding = torch.load(file_path)
                embeddings.append(embedding)
                # Assuming the document ID is the filename without the '.pt' extension
                doc_id = file[:-3]
                doc_ids.append(doc_id)  

        if embeddings:
            self.embeddings = torch.stack(embeddings)
            self.doc_ids = doc_ids

    def vectorize_query(self, query):
        # Tokenize the query
        tokens = word_tokenize(query)
        
        # Compute the embedding for each token
        token_embeddings = [self.model.encode(token, convert_to_tensor=True) for token in tokens]
        
        # Compute the embedding for the entire query
        entire_query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Add the entire query embedding to the list of token embeddings
        token_embeddings.append(entire_query_embedding)
        
        # Stack them into a single tensor
        query_embedding = torch.stack(token_embeddings)
        
        return query_embedding



    def retrieve_top_n(self, query, k=10):
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
        
        sum_over_docs = self.compute_scores(query_vector, k)

        # Ensure top_k does not exceed the number of documents
        adjusted_top_k = min(k, sum_over_docs.size(0))
        scores, indices = torch.topk(sum_over_docs, k=adjusted_top_k)

        # Assuming self.doc_ids stores document IDs corresponding to the indices in self.bert_embeddings
        doc_ids = [self.doc_ids[idx] for idx in indices]
    
        ranked_docs = list(zip(doc_ids, scores))

        
        return ranked_docs

    def compute_scores(self, query_vector, top_k=10):
        sim_scores_per_doc = []
        epsilon = 1e-10  # Small value to avoid division by zero

        print("Embedding",self.embeddings.shape)
        for query_embedding in query_vector:
            print("Query Embed",query_embedding.shape)
            # Normalize query_embedding
            query_embedding_norm = query_embedding / (torch.norm(query_embedding, p=2, dim=0, keepdim=True) + epsilon)
            
            # Normalize bert_embeddings
            bert_embeddings_norm = self.embeddings / (torch.norm(self.embeddings, p=2, dim=2, keepdim=True) + epsilon)
            
            # Perform dot product using matmul, now with normalized embeddings
            scores = torch.matmul(bert_embeddings_norm, query_embedding_norm.T)  # Transpose query_embedding_norm for matmul
            print(scores.shape)
            max_scores, _ = torch.max(scores, dim=1)
            sim_scores_per_doc.append(max_scores)
        
        # Convert list to tensor
        sim_scores_tensor = torch.stack(sim_scores_per_doc)
        # Sum over the documents
        sum_over_docs = torch.sum(sim_scores_tensor, dim=0)

        return sum_over_docs
    
    def get_top_k(self, scores, top_k):
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    