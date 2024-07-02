import numpy as np
import re
import nltk
import math
import json
import sys
import os

from pathlib import Path
from collections import Counter
from nltk.corpus import reuters, stopwords
from nltk.stem import PorterStemmer

sys.path.insert(0, Path(__file__).resolve().parents[1])

class Index():
    
    def __init__(self, corpus):
        """
        A class to create and export index information given a corpus.

        This method is responsible for any preprocessing of the documents 
        and storing the necessary information for ranking.
        
        Args:
            corpus (dict of str: str): Dictionary of documents, where each document has
                the following structure:
                    {doc_id: text}    
        """
        
        self.corpus = corpus
        self.doc_ids = corpus.keys()
        
        # nltk intializations
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        
        self.docs_total = len(corpus)
    
    def initialize_index(self):

        doc_lengths = self.document_lengths()
        
        tfs = self.tf(self.corpus)
        idfs = self.idf(self.corpus)
        # tfidfs = self.tfidf(self.corpus)
        
        self.index_data = {doc_id: {
            'length': doc_lengths[doc_id],
            
            'tf': tfs[doc_id],
            'idf': idfs,
            # 'tfidf': tfidfs[doc_id],
            
            # 'word2vec': self.embedding(corpus[doc_id]),
            # ...
            } for doc_id in self.doc_ids}
          
    
    def bag_of_words(self, text):
        """
        Extracts bag of words from the given text.

        Args:
            text (str): The text from which to extract bag of words.

        Returns:
            list: List of unique words in the text.
        """
        text = re.sub(r'[^\w\s]', '', text)
        # tokenize the text
        tokens = text.lower().split()
        # apply stemming and rem. stopwords
        tokens = [self.stemmer.stem(token) for token in tokens
                  if token not in self.stopwords]
        
        return tokens
    
    def document_lengths(self):
        """
        Calculates the document lengths.

        Returns:
            dict: Dictionary of document lengths.
        """
        return {doc_id: len(self.bag_of_words(doc)) for doc_id, doc in self.corpus.items()}
    
    def tf(self, docs):
        """Term frequency of the documents in the given corpus.

        Args:
            docs (dict): Dictionary of documents represented as bag of words.
            
        Returns:
            dict: Dictionary of term frequencies for each document.
        """
        return {doc_id: Counter(self.bag_of_words(doc)) for doc_id, doc in docs.items()}
    
    def idf(self, docs):
        """
        Calculates the inverse document frequency (IDF) for each unique word in the given corpus.

        Args:
            docs (dict): Dictionary of documents, where each document is represented as a list of words.

        Returns:
            dict: Dictionary of IDF values for each unique word in the entire corpus of documents.
        """
        N = self.docs_total  # Total number of documents
        df = {}  # Document frequencies

        # Calculate document frequency for each word
        for doc in docs.values():
            unique_words = set(self.bag_of_words(doc)) # Consider each word once per document
            for word in unique_words:
                df[word] = df.get(word, 0) + 1

        # Calculate IDF for each word (added laplace smoothing for unseen words in the corpus)
        idf_values = {word: math.log((N+1) / (df[word]+1)) for word in df}

        return idf_values
    
    def embedding(self, docs):
        """
        

        Args:
            docs (_type_): _description_
        """
    
    def export_index(self, index_name):
        """
        Exports the index to the specified file path.
        
        Structure of the index (json) is:
        {
            'doc_id': {
                'length': len of document,
                
                'tf': tf_values,
                'idf': idf_values,
                
                'word2vec': word2vec_embedding of doc,
                ...
                }
        }

        Args:
            path (str): The file path where to export the inverted index.
        """
        path = Path("dat", f"{index_name}.json")
        
        with open(path, 'w') as f:
            json.dump(self.index_data, f, indent=4)
    
    
    
class QueryLikelihoodModel():
    
    def __init__(self, corpus):
        """
        Initialize Query Likelihood Model on given corpus of documents.
        
        Args:
            corpus (dict): Dictionary of documents, where each document has
                the following structure:
                    {'doc_id': str, 'text': str}
        Raises:
            AssertionError: if the corpus does not have the correct structure        
        """
        super().__init__(corpus)
        
        self.ranker = 'query_likelihood'        
    
    def rank(query):
        pass
    



class BM25():
    
    def __init__(self, index_path):
        """
        Initialize BM25 on given corpus of documents.
        
        Args:
            corpus (dict): Dictionary of documents, where each document has
                the following structure:
                    {'doc_id': str, 'text': str}
        Raises:
            AssertionError: if the corpus does not have the correct structure        
        """
        self.ranker = 'bm25'
        
        # nltk intializations
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        
        # Load index
        with open(index_path, 'r') as file:
            index_data = json.load(file)
            
        self.doc_ids = index_data.keys()
            
        self.tf = {doc_id: index_data[doc_id]['tf'] for doc_id in index_data}
        self.idf = {doc_id: index_data[doc_id]['idf'] for doc_id in index_data}
    
    def tfidf(self, query, doc_id):
        """
        Calculates the TF-IDF score for a query on the corpus provided inside the class.
        
        Args:
            query (str): The query string for which to calculate the TF-IDF scores.
        """
        
        query_tokens = self.bag_of_words(query)
        
        tfidf_scores = []
        for query_token in query_tokens:
            query_idf = self.idf[doc_id][query_token] # TODO: Whaat if query_token is not in idf -> smoothing?
            query_tf = self.tf[doc_id].get(query_token, 0)
            tfidf_scores.append(query_tf * query_idf)
        
        return sum(tfidf_scores)
    
    def rank_tfidf(self, query):
        """
        Ranks documents using TF-IDF scores.
        
        Args:
            query (str): The query string for which to calculate the TF-IDF scores.

        Returns:
            list: List of document IDs and their corresponding TF-IDF scores.
        """
        self.ranks = {doc_id: 0 for doc_id in self.doc_ids}
        
        for doc_id in self.doc_ids:
            tfidf_score = self.tfidf(query, doc_id)
            self.ranks[doc_id] = tfidf_score
            
        return sorted(self.ranks.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def bag_of_words(self, text):
        """
        Extracts bag of words from the given text.

        Args:
            text (str): The text from which to extract bag of words.

        Returns:
            list: List of unique words in the text.
        """
        text = re.sub(r'[^\w\s]', '', text)
        # tokenize the text
        tokens = text.lower().split()
        # apply stemming and rem. stopwords
        tokens = [self.stemmer.stem(token) for token in tokens
                  if token not in self.stopwords]
        
        return tokens
    
    def rank(query):
        pass
    
    
    
if __name__ == "__main__":
    
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
    
    index = Index(corpus)
    index.initialize_index()
    
    index_name = 'test_index'
    index.export_index('test_index')
    
    
    index_path = Path("dat", f"{index_name}.json")

    # Initialize BM25
    bm25 = BM25(index_path)
    ranked_docs = bm25.rank_tfidf('french bulldog')
    print(ranked_docs)