import numpy as np
import re
import math
import json
import sys
import os
import csv

from pathlib import Path
from collections import Counter
from nltk.corpus import reuters, stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix

sys.path.insert(0, Path(__file__).resolve().parents[1])


class TextEmbedding():
    
    def __init__(self, corpus=None):
        """
        A class to create text embeddings.

        """
        self.corpus = corpus
        
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
    
    def get_doc_ids(self):
        return self.corpus.keys()
    
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
        
        return ' '.join(tokens)
    
    # word2vec embeddings
    def vectorize(self, text):
        """
        Vectorizes the given text using word embeddings.

        Args:
            text (str): The text to vectorize.

        Returns:
            np.array: The vectorized representation of the text.
        """
        pass # TODO: Immplement
    
    
    # ? PageRank calculations somewhere else I guess, use adjacentcy matrix?
    def incoming_pages(self, doc_id):
        """
        Returns the incoming pages of the given document.

        Args:
            doc_id (str): The document ID for which to find incoming pages.

        Returns:
            list: List of incoming pages.
        """
        pass # TODO: Implement somewhere eles, does not fit in here.


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
        self.text_embedding = TextEmbedding(corpus)
        
        self.docs_total = len(corpus)
        self.doc_ids = self.text_embedding.get_doc_ids()
        
        self.tf_vectorizer = CountVectorizer()
        self.tfidf_tranformer = TfidfTransformer()
    
    def initialize_index(self, from_scratch=False):
        
        print("Initializing index...")
        
        doc_ids = list(self.doc_ids)
        
        tfs, token_names = self.tf(self.corpus)
        tfidfs = self.tfidf(tfs)
        doc_lengths = self.doc_lens(tfs)
        
        self.index_data = {
            'doc_ids': doc_ids,
            'doc_lengths': doc_lengths,
            'token_names': token_names,
            'tfs': tfs,
            'tfidfs': tfidfs,
        }

        print("Index initialized successfully.")
        
        if not from_scratch:
            self.index_data_doc_id = None
            return
        
        # Calculate TF and IDF from scratch, if from_scratch is True
        tfs_from_scratch = self.tf_from_scratch(self.corpus)
        idfs_from_scratch = self.idf_from_scratch(self.corpus)
        doc_lengths = self.document_lengths()
        
        self.index_data_doc_id = {doc_id: {
            'length': doc_lengths[doc_id],
            'tf': tfs_from_scratch[doc_id],            
            } for doc_id in self.doc_ids}
        self.index_data_doc_id.update({'idfs': idfs_from_scratch})
        
    
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
        feature_names = self.tf_vectorizer.get_feature_names_out()
        
        return X, feature_names.tolist()
    
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
    
    ## * ###### From scratch calculations ###### * ##
    def document_lengths(self):
        """
        Calculates the document lengths for each doc_id given len of bow.

        Returns:
            dict: Dictionary of document lengths.
        """
        return {
            doc_id: len(self.text_embedding.bag_of_words(doc).split(' ')) 
            for doc_id, doc in self.corpus.items()
            }
        
    def tf_from_scratch(self, docs):
        """Term frequency of the documents in the given corpus.

        Args:
            docs (dict): Dictionary of documents represented as bag of words.
            
        Returns:
            dict: Dictionary of term frequencies for each document.
        """
        return {
            doc_id: Counter(self.text_embedding.bag_of_words(doc).split(' ')) 
            for doc_id, doc in docs.items()
            }
    
    def idf_from_scratch(self, docs):
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
            unique_words = set(self.text_embedding.bag_of_words(doc).split(' ')) # Consider each word once per document
            for word in unique_words:
                df[word] = df.get(word, 0) + 1

        # Calculate IDF for each word (added laplace smoothing for unseen words in the corpus)
        idf_values = {word: math.log((N+1) / (df[word]+1)) for word in df}

        return idf_values
    ## * ###### From scratch calculations ###### * ##
    
    def export_index(self, index_name):
        """
        Exports the index to the specified file path.
        
        Structure of index_data_doc_id is:
        {
            doc_id: {'length': len of document, 'tf': tf_values},
            'idfs': idf_values
        }
        Structure of inddex_data is:
        {
            'doc_ids': list of doc_ids,
            'doc_lengths': list of doc_lengths,
            'tfs': tf_values,
            'tfidfs': tfidf_values
        }

        Args:
            index_name (str): Name of the index to export. 
        """
        def dump_json(path, data):
            def default(obj):
                if isinstance(obj, csr_matrix):
                    # Convert csr_matrix to a dictionary that holds non-zero elements and their indices
                    return {
                        "type": "csr_matrix", 
                        "data": obj.data.tolist(), 
                        "indices": obj.indices.tolist(), 
                        "indptr": obj.indptr.tolist(), 
                        "shape": obj.shape
                    }
                raise TypeError("Object of type %s is not JSON serializable" % type(obj).__name__)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=4, default=default)
        
        if self.index_data_doc_id is not None:
            path = Path("dat", f"{index_name}_by_id.json")
            dump_json(path, self.index_data_doc_id)
            
        path = Path("dat", f"{index_name}.json")
        dump_json(path, self.index_data)
        
        print(f"Index exported to {path}.")
        


class BM25():
    
    def __init__(self, index_path):
        """
        Initialize BM25 on given corpus of documents.
        
        Args:
            path (str): The file path to the index.   
        """
        self.ranker = 'bm25'
        
        self.index_path = index_path
        self.text_embedding = TextEmbedding()
        
    
    @staticmethod
    def load_csr_matrix(data):
        if data['type'] == 'csr_matrix':
            return csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
        else:
            raise ValueError("JSON does not contain csr_matrix data")

    def load_index(self, by_doc_id=False):
        
        with open(self.index_path, 'r') as file:
            index_data = json.load(file)
        
        if by_doc_id:
            # The slow from scratch implementation
            self.idf = index_data.pop('idfs')
            self.doc_ids = index_data.keys()
            self.doc_lengths = {doc_id: index_data[doc_id]['length'] for doc_id in index_data}
            self.tf = {doc_id: index_data[doc_id]['tf'] for doc_id in index_data}
        
        else:
            # The fast implementation
            self.doc_ids = index_data['doc_ids']
            self.doc_lengths = index_data['doc_lengths']
            self.token_names = index_data['token_names']
            self.tf = BM25.load_csr_matrix(index_data['tfs'])
            self.idf = None
            self.tfidf = BM25.load_csr_matrix(index_data['tfidfs'])
        
    ## * ######## The slow tfidf at query time implementation (lecture slides) ######## * ##
    def tfidf(self, query, doc_id):
        """
        Calculates the TF-IDF score for a query on the corpus provided inside the class.
        
        #! This is the slow implementation of the TF-IDF score calculation. Based on the lecture slides.
        
        Args:
            query (str): The query string for which to calculate the TF-IDF scores.
        """
        if self.idf is None:
            raise Exception("No IDF scores found. Load the index with by_doc_id=True.")
        
        query_tokens = self.text_embedding.bag_of_words(query).split(' ')
        
        tfidf_scores = []
        for query_token in query_tokens:
            query_idf = self.idf[query_token] # TODO: Whaat if query_token is not in idf -> smoothing?
            query_tf = self.tf[doc_id].get(query_token, 0)
            tfidf_scores.append(query_tf * query_idf)
        
        return sum(tfidf_scores)
    
    def rank_tfidf(self, query):
        """
        Ranks documents using TF-IDF scores.
        
        #! This is the slow implementation of the ranking. Based on the lecture slides.
        
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
    ## * ######## The slow tfidf at query time implementation (lecture slides) ######## * ##
    
    def vectorize_query(self, query):
        
        query_tokens = self.text_embedding.bag_of_words(query).split(' ')
        token_index = {token: i for i, token in enumerate(self.token_names)}
        
        # create a query vector out of query tokens
        query_vec = np.zeros(len(self.token_names))
        for token in query_tokens:
            index = token_index.get(token)
            if index is not None:
                query_vec[index] = 1
                
        return query_vec
    
    def rank(self, query):
        
        query_vector = self.vectorize_query(query)
        scores = self.tfidf.dot(query_vector)
        
        ranked_documents = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        return [(self.doc_ids[doc_id], score) for doc_id, score in ranked_documents]
    

def load_corpus_from_json_files(directory_path):
    corpus = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Assuming the text content you want is under a key named 'text'.
                # Adjust the key according to the actual structure of your JSON files.
                corpus[filename[:-5]] = data['text']
    return corpus

def load_url_mapping_from_csv(csv_file_path):
    url_mapping = {}
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        header = reader.fieldnames
        print("CSV Header:", header)
        
        for row in reader:
            doc_id, url = row[header[0]], row[header[1]]
            url_mapping[doc_id] = url
            
    return url_mapping
    
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
    index.initialize_index(from_scratch=True)
    
    index_name = 'test_index'
    index.export_index('test_index')
    
    # Lecture implementation of tfidf ranker class
    index_path_by_id = Path("dat", f"{index_name}_by_id.json")
    bm25_slow = BM25(index_path_by_id)
    bm25_slow.load_index(by_doc_id=True)
    ranked_docs = bm25_slow.rank_tfidf('french bulldog')
    print(ranked_docs)
    
    # tfidf ranker class using sklearn fit transform
    index_path = Path("dat", f"{index_name}.json")
    bm25 = BM25(index_path)
    bm25.load_index(by_doc_id=False)
    ranked_docs = bm25.rank('french bulldog')
    print(ranked_docs)
    
    
    # Try on crawled tuebingen data
    directory_path = "dat/crawler_test_run/crawler/index/"
    corpus_tue = load_corpus_from_json_files(directory_path)
    print(len(corpus_tue))
    print(list(corpus_tue.items())[:2], end='\n\n')
    
    mapping_path = "dat/crawler_test_run/crawler/20240703010421.csv"
    url_mapping = load_url_mapping_from_csv(mapping_path)
    print(len(url_mapping))
    print(list(url_mapping.items())[:5], end='\n\n')
    
    index_tue = Index(corpus_tue)
    index_tue.initialize_index(from_scratch=False)
    
    index_name_tue = 'tuebingen_index'
    index_tue.export_index(index_name_tue)
    
    index_path_tue = Path("dat", f"{index_name_tue}.json")
    bm25_tue = BM25(index_path_tue)
    bm25_tue.load_index(by_doc_id=False)
    ranked_docs_tue = bm25_tue.rank('gastronomie in tübingen restaurant')
    
    top5 = ranked_docs_tue[:5]
    print("Top 5 documents for 'gastronomie in tübingen restaurant':")
    for doc_id, score in top5:
        print(f"Document ID: {doc_id}, Score: {score}, URL: {url_mapping.get(doc_id)}")