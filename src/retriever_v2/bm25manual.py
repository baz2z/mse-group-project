from pathlib import Path
from pickle import dump, load

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import math
import collections

from retriever_v2.base import BaseRetriever, Document, RetrievalScore
from retriever_v2.utils import PICKLES_DIR, STOPWORDS, TOKEN_PATTERN

STEMMER = SnowballStemmer("english")


def stem_tokenize(text: str):
    return [
        STEMMER.stem(word.lower())
        for word in word_tokenize(text)
        if word.lower() not in STOPWORDS and TOKEN_PATTERN.match(word)
    ]
def documents_to_dict(documents: list[Document]) -> dict:
    return {doc.doc_id: doc.text for doc in documents}

class BM25ManualRetriever(BaseRetriever):
    def __init__(
        self,
        pickles_dir: Path = PICKLES_DIR,
        documents: list[Document] = None,
        k1: float = 2,
        b: float = 1,
  
    ):
        self.ids = [doc.doc_id for doc in documents]

        try:
            with open(pickles_dir / "bm25manual.pkl", "rb") as f:
                self.bm25 = load(f)
        except FileNotFoundError:
            if not documents:
                raise ValueError("No documents provided and no index found")

            dicts_for_legacy_bm25 = documents_to_dict(documents)
            self.bm25 = self.precompute(dicts_for_legacy_bm25, pickles_dir, k1, b)

    def score(self, query: str) -> list[RetrievalScore]:
        # if not (query_tokenized := stem_tokenize(query)):
        #     return []

        scores = self.bm25.get_scores(query, self.ids)
        return [
            RetrievalScore(doc_id=doc_id, score=score, ranker="bm25")
            for doc_id, score in scores
        ]

    @staticmethod
    def precompute(
        documents: list[Document],
        pickles_dir: Path = PICKLES_DIR,
        k1: float = 2,
        b: float = 1,
    ):
        
        bm25 = BM25(documents, k1=k1, b=b)

            
        with open(pickles_dir / "bm25manual.pkl", "wb") as f:
            dump(bm25, f)

        return bm25



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
    def __init__(self, corpus, k1, b):
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
        self.embed = BagOfWordsTokenizer(corpus=corpus)

        self.doc_embeddings = self.embed.tokenize_corpus()       
        # Pre compute tf and idf scores
        self.initialize()

    @property
    def corpus_size(self):
        """
        Number of documents in the corpus.
        """
        return len(self.doc_len)
    


    def initialize(self):
        """
        Calculates frequencies of terms in documents and in corpus. 
        Also computes inverse document frequencies.
        """ 
        self.compute_term_frequencies(self.doc_embeddings)
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
        
    def get_scores(self, query, doc_ids):
        """
        Retrieve the top n documents for the query.

        Args:
            query (list of str): The tokenized query
            documents (list): The documents to return from (doc_id with the same order as the corpus)
            n (int): The number of documents to return

        Returns:
            list of tupes: The top n documents of the form (doc_id, score)
        """
        query = self.embed.tokenize(query)
        assert self.corpus_size == len(doc_ids), \
            "The documents given don't match the index corpus!"
        
        scores = collections.defaultdict(float)
        for token in query:
            if token in self.tf:
                for index, freq in self.tf[token].items():
                    norm_doc_len = self.k1 * \
                        (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
                    scores[index] += self.idf[token] * freq * (self.k1 + 1) / (freq + norm_doc_len)
        
        return [(doc_ids[i], scores[i]) for i in scores]
class BagOfWordsTokenizer():
    """
    A class to create bag of words tokens from a given corpus.
    """
    def __init__(self, corpus):
        """
        Init tokenizer with given corpus and load stemmer and stopwords.
                
        Args:
            corpus (dict): A dictionary where the keys are document IDs and the values are the
                corresponding documents.
        """
        self.corpus = corpus
        
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))

    @property
    def doc_ids(self):
        """
        Ordered document IDs in the corpus.
        """
        return list(self.corpus.keys())
    
    def tokenize_corpus(self):
        """
        Tokenizes the corpus by extracting bag of words from each document.

        Returns:
            list of list of strings: Tokenized documents.
        """
        tokenized_docs = []
        for doc in tqdm(self.corpus.values(), desc="Tokenize documents"):
            tokenized_doc = self.tokenize(doc)
            tokenized_docs.append(tokenized_doc)
        return tokenized_docs
    
    def tokenize(self, text):
        """
        Extracts bag of words tokens from the given text.

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

