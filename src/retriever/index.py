import numpy as np
import math
import json
import sys

from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix

sys.path.insert(0, Path(__file__).resolve().parents[1])

# internal imports
from text_embedding import TextEmbedding
from in_out import convert_csr_to_dict

class Index():
    """
    A class to create and export index information given a corpus.
    """
    
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
        self.text_embedding = TextEmbedding(corpus)
        
        self.docs_total = len(corpus)
        self.doc_ids = self.text_embedding.get_doc_ids()
        
        self.tf_vectorizer = CountVectorizer()
        self.idf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
        self.tfidf_tranformer = TfidfTransformer(smooth_idf=True)

        # self.bert_embeddings = self.text_embedding.get_bert_embeddings()


    def initialize_index(self):
        
        print("Initializing index...")
        
        doc_ids = list(self.doc_ids)
        
        tfs, token_names = self.tf(self.corpus)
        idfs = self.idf(tfs, token_names)
        tfidfs = self.tfidf(tfs)
        doc_lengths = self.doc_lens(tfs)

        # bert_embeddings = self.bert_embeddings
        
        self.index_data = {
            'doc_ids': doc_ids,
            'doc_lengths': doc_lengths,
            'token_names': token_names,
            'tfs': tfs,
            'idfs': idfs,
            'tfidfs': tfidfs,
            # 'bert_embeddings': bert_embeddings,
        }

        print("Index created successfully.")
    
    def tf(self, docs):
        """
        Term frequency of the documents in the given corpus.

        Args:
            docs (dict): Dictionary of the given corpus

        Returns:
            array of shape (n_samples, n_features): Document-term matrix.
        """
        preprocessed_docs = [self.text_embedding.bag_of_words(doc) for doc in docs.values()]
        
        # Create term frequency matrix
        X = self.tf_vectorizer.fit_transform(preprocessed_docs)
        # Columns of the term frequency matrix (all unique tokens in the corpus)
        token_names = self.tf_vectorizer.get_feature_names_out()
        
        return X, token_names.tolist()
    
    def doc_lens(self, tfs):
        """
        Returns the number of unique words for each document based on the tf vectorizer.

        Args:
            tfs (_type_): array of shape (n_samples, n_features): Document-term matrix.

        Returns:
            _type_: 1D array of shape (n_samples,): Number of unique words for each document.
        """
        binary = tfs.copy()
        binary[binary > 0] = 1
        
        return np.array(binary.sum(axis=1)).flatten().tolist()
    
    def idf(self, tfs, token_names):
        """
        Calculates the inverse document frequency given the output of the 
        CountVectorizer fit_transform method.

        Args:
            tfs (_type_): array of shape (n_samples, n_features): Document-term matrix.

        Returns:
            dict: Dictionary of unique words and their corresponding idf values.
        """
        self.idf_transformer.fit(tfs)
        idf_values = self.idf_transformer.idf_
        
        return dict(zip(token_names, idf_values))
    
    def tfidf(self, tfs):
        """
        Calculates the tf-idf representation of the given documents based on the 
        tf vectorizer.

        Args:
            tfs (_type_): array of shape (n_samples, n_features): Document-term matrix.
        
        Returns:
            _type_: array of shape (n_samples, n_features): Document-term matrix with tf-idf values.
        """
        return self.tfidf_tranformer.fit_transform(tfs)
    
    def export_index(self, index_name):
        """
        Exports the index to the specified file path.
        
        Structure of index_data is:
        {
            'doc_ids': list of doc_ids,
            'doc_lengths': list of doc_lengths,
            'tfs': tf_values,
            'tfidfs': tfidf_values
        }

        Args:
            index_name (str): Name of the index to export. 
        """
        def dump_to_json(path, data):
            with open(path, 'w') as f:
                json.dump(
                    data, f, indent=4, 
                    default=convert_csr_to_dict
                    )
          
        path = Path("dat", f"{index_name}.json")
        dump_to_json(path, self.index_data)
        
        print(f"Index exported to {path}.")
        
