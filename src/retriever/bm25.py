"""
BM25 ranking algorithm.
"""

import collections
import heapq
import math
import pickle
import sys
import os

PARAM_K1 = 1.5
PARAM_B = 0.75
IDF_CUTOFF = 0

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class BM25:
    """
    Best Matching 25 ranking function

    Attributes:
        tf (dict of token: <doc, freq>): Dictionary with terms frequencies for each 
            document in corpus.
        idf (dict of token: idf score): Pre computed IDF score for every term.
        doc_len (list of int): List of document lengths.
        avgdl (float): Average length of document in `corpus`.
    """
    def __init__(self, corpus, k1=PARAM_K1, b=PARAM_B):
        """
        Args
            corpus (list of list of str): Given corpus.
            k1 (float): Constant used for influencing the term frequency saturation.
            b (float): Constant used for influencing the effects of different document 
                lengths relative to average document length.
        """
        # Initialize BM25 parameters
        self.k1 = k1
        self.b = b

        # Initialize attributes
        self.avgdl = 0
        self.tf = {}
        self.idf = {}
        self.doc_len = []
                
        # Pre compute tf and idf scores
        self.initialize(corpus)

    @property
    def corpus_size(self):
        """
        Number of documents in the corpus.
        """
        return len(self.doc_len)

    def initialize(self, corpus):
        """
        Calculates frequencies of terms in documents and in corpus. 
        Also computes inverse document frequencies.
        """
        self.compute_term_frequencies(corpus)
        self.compute_average_document_length()
        self.compute_inverse_document_frequencies()


        self.average_idf = sum(self.idf.values()) / len(self.idf)
        if self.average_idf < 0:
            print(
                'Average inverse document frequency is less than zero.'
                f'Your corpus of {self.corpus_size} documents'
                ' is either too small or it does not originate from natural text. BM25 may produce'
                ' unintuitive results.'
            )

    def compute_term_frequencies(self, corpus):
        for i, document in enumerate(corpus):
            self.doc_len.append(len(document))

            for word in document:
                if word not in self.tf:
                    self.tf[word] = {}
                if i not in self.tf[word]:
                    self.tf[word][i] = 0
                self.tf[word][i] += 1

    def compute_inverse_document_frequencies(self):
        for word, docs in self.tf.items():
            idf = math.log(self.corpus_size - len(docs) + 0.5) - math.log(len(docs) + 0.5)
            self.idf[word] = idf

    def compute_average_document_length(self):
        self.avgdl = sum(self.doc_len)/len(self.doc_len)

    def save(self, filename):
        full_path = os.path.join(BASE_DIR, 'retriever', filename)
        print(f"path bm25 gets dumped in: {os.path.abspath(full_path)}")
        with open(f"{full_path}.pkl", "wb") as fsave:
            pickle.dump(self, fsave, protocol=pickle.HIGHEST_PROTOCOL)
                
    ### 
    # Below methods are used for loading the BM25 class with precomputed tf and idf scores
    # and for retrieving the top n documents for a given query. Only the top n scores are 
    # computed and returned.
    ###
    @staticmethod
    def load(filename):
        with open(f"{filename}.pkl", "rb") as fsave:
            return pickle.load(fsave)
        
    def retrieve_top_n(self, query, documents, n=5):
        """
        Retrieve the top n documents for the query.

        Args:
            query (list of str): The tokenized query
            documents (list): The documents to return from (doc_id with the same order as the corpus)
            n (int): The number of documents to return

        Returns:
            list of tupes: The top n documents of the form (doc_id, score)
        """
        assert self.corpus_size == len(documents), \
            "The documents given don't match the index corpus!"
        
        scores = collections.defaultdict(float)
        for token in query:
            if token in self.tf:
                for index, freq in self.tf[token].items():
                    norm_doc_len = self.k1 * \
                        (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
                    scores[index] += self.idf[token] * freq * (self.k1 + 1) / (freq + norm_doc_len)

        return [
            (documents[i], scores[i]) 
            for i in heapq.nlargest(n, scores.keys(), key=scores.__getitem__)
            ]