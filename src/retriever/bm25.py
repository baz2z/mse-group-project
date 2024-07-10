import numpy as np
import math
import json
import sys

from tqdm import tqdm
from pathlib import Path
from scipy.sparse import csr_matrix

sys.path.insert(0, Path(__file__).resolve().parents[1])

# internal imports
from text_embedding import TextEmbedding
from in_out import load_csr_matrix


class BM25():
    """
    BM25 (Best Matching 25) is a ranking algorithm used for information retrieval that 
    computes relevance scores for documents given a query based on the tfidf approach.
    """
    
    def __init__(self, index_path=None, index_name=None):
        """
        Initialize BM25 on given pre computed index.
        
        Args:
            path (str): The file path to the index.   
        """
        self.ranker = 'bm25'
        self.text_embedding = TextEmbedding()
       
        # Initialize index
        self.index_path = index_path 
        self.index_name = index_name       
        self.initialize_index_numpy()
        
        # Initialize BM25 parameters
        self.N = len(self.doc_ids)
        self.avg_doc_len = sum(self.doc_lengths) / self.N
        
        self.k = 1.5  # default k in range [1.2, 2.0]
        self.b = 0.75 # default value for b
        
        print("BM25 initialized")

    def initialize_index(self):
        """
        Initializes the index by loading the necessary data from the index json file.
        """
        with open(self.index_path, 'r') as file:
            index_data = json.load(file)
        
        self.doc_ids = index_data['doc_ids']
        self.token_names = index_data['token_names']
        self.doc_lengths = np.array(index_data['doc_lengths'])
        self.tf = load_csr_matrix(index_data['tfs'])
        # self.idf = index_data['idfs']
        self.tfidf = load_csr_matrix(index_data['tfidfs'])

    def initialize_index_numpy(self):
        """
        Load the index from the numpy files.
        
        Loads the index desc from the npz file given the index name.
        Yields tf and tfidf matrices after loading their respective npy file given
        the index name.
        """
        index_desc_path = Path("dat", f"{self.index_name}_desc.npz")
        index_desc = np.load(index_desc_path, allow_pickle=True)
        self.doc_ids = index_desc['doc_ids']
        self.token_names = index_desc['token_names']
        self.doc_lengths = np.array(index_desc['doc_lengths'])
        
        self.index_tf_dir = Path("dat", f"{self.index_name}_tfs")        
        self.index_tfidf_dir = Path("dat", f"{self.index_name}_tfidfs")
        
    def load_matrix(self):
        """
        Generator to loop through all npy files inside the index_tf_dir and index_tfidf_dir.
        Yields a pair of tf and tfidf arrays.
        """
        if not self.index_tf_dir.exists():
            raise FileNotFoundError(f"Folder not found: {self.index_tf_dir}")
        
        for tf_file_path in self.index_tf_dir.glob('*.npy'):
            tfidf_file_name = tf_file_path.name.replace('_tfs', '_tfidfs')
            tfidf_file_path = self.index_tfidf_dir / tfidf_file_name
            indices = tfidf_file_name.split('.')[0].split('_')
            
            if tfidf_file_path.exists():
                tf_matrix = np.load(tf_file_path)
                tfidf_matrix = np.load(tfidf_file_path)
                
                yield (indices, tf_matrix, tfidf_matrix)
        
        

    def vectorize_query(self, query):
        """
        Vectorizes the given query by creating a query vector based on the tokens in the query.

        Args:
            query (str): The query string.

        Returns:
            numpy.ndarray: The query vector representing the query.

        """
        query_tokens = self.text_embedding.bag_of_words(query).split(' ')
        token_index = {token: i for i, token in enumerate(self.token_names)}
        
        # create a query vector out of query tokens
        query_vec = np.zeros(len(self.token_names))
        for token in query_tokens:
            index = token_index.get(token)
            if index is not None:
                query_vec[index] = 1
                
        return query_vec
    
    def rank_tfidf(self, query):
        """
        Ranks documents using TF-IDF scores.
        
        Args:
            query (str): The query string for which to calculate the TF-IDF scores.

        Returns:
            list: A list of tuples, where each tuple contains a document ID and its relevance score.
        """
        query_vector = self.vectorize_query(query)
        scores = self.tfidf.dot(query_vector)
        
        ranked_documents = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        
        return [
            (self.doc_ids[doc_id], score) 
            for doc_id, score in ranked_documents
        ]
    
    # TODO: vectorize for better performance
    def rank(self, query, top_k=100):
        """
        BM25 ranking algorithm. 
        
        Uses the tfidf approach to compute relevance scores for documents given a query. 
        
        Args:
            query (str): The query string.
            
        Returns:
            list: A list of tuples, where each tuple contains a document ID and its relevance score.
        """
        query_vector = self.vectorize_query(query)
        scores = self.compute_scores_numpy(query_vector)            
        print("After scoring")
        ranked_documents = self.get_top_k(scores, top_k)

        return ranked_documents
    
    def compute_scores(self, query_vector):
        """
        Compute BM25 scores for each document given a query vector.
        
        Args:
            query_vector (numpy.ndarray): The query vector.
        
        Returns:
            numpy.ndarray: BM25 scores for each document.
        """
        scores = np.zeros(len(self.doc_ids))
        
        for i, doc_id in tqdm(enumerate(self.doc_ids)):
            doc_length = self.doc_lengths[i]
            for j, term in enumerate(self.token_names):
                if query_vector[j] > 0:
                    idf = self.idf[term]
                    tf_qi_D = self.tf[i, j]
                    term_score = idf * (tf_qi_D * (self.k+1)) / \
                        (tf_qi_D + self.k * (1 - self.b + self.b * (doc_length / self.avg_doc_len)))
                    scores[i] += term_score
        return scores
    
    def compute_scores_fast(self, query_vector):
        
        query_vector_sparse = csr_matrix(query_vector)
        
        # Pre-compute document length normalization factor -> array of size len(docs)
        doc_length_norm = self.k * (1 - self.b + self.b * (self.doc_lengths / self.avg_doc_len))
        
        print("before tfidf compute")
        # Compute term scores using vectorized operations
        term_scores = self.tfidf.multiply(query_vector_sparse)
        print("after tfidf compute")
        term_scores = term_scores.multiply(self.k + 1) / (self.tf + doc_length_norm[:, None])
        print("normalized term scores")
        # Sum scores across terms for each document
        scores = term_scores.sum(axis=1).A1
        print("sum")
        return scores
    
    def compute_scores_fast_batch(self, query_vector, batch_size=1000):
        """
        Compute BM25 scores for each document given a query vector using batching to reduce memory allocation.
        
        Args:
            query_vector (numpy.ndarray): The query vector.
            batch_size (int): The number of documents to process in each batch.
        
        Returns:
            numpy.ndarray: BM25 scores for each document.
        """
        query_vector_sparse = query_vector
        scores = np.zeros(len(self.doc_ids))
        
        num_batches = math.ceil(len(self.doc_ids) / batch_size)
        
        print("Num batches: %d" % num_batches)
        for batch_idx in tqdm(range(num_batches), desc="Computing scores"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.doc_ids))
            
            tfidf_batch = self.tfidf[start_idx:end_idx].toarray()
            tf_batch = self.tf[start_idx:end_idx].toarray()
            print("tfidf_batch shape: ", tfidf_batch.shape)
            print("tf_batch shape: ", tf_batch.shape)
            print("query_vector_sparse shape: ", query_vector_sparse.shape)
            
            term_scores_batch = tfidf_batch * query_vector_sparse
            doc_length_norm_batch = self.k * (1 - self.b + self.b * (self.doc_lengths[start_idx:end_idx] / self.avg_doc_len))
            
            term_scores_batch = (term_scores_batch * (self.k + 1)) / (tf_batch + doc_length_norm_batch[:, None])
            
            scores_batch = term_scores_batch.sum(axis=1).flatten()
            scores[start_idx:end_idx] = scores_batch
        
        return scores
    
    def compute_scores_numpy(self, query_vector):
        scores = np.zeros(len(self.doc_ids))
        
        for (indices, tf_matrix, tfidf_matrix) in self.load_matrix():
            start_idx, end_idx = int(indices[0]), int(indices[1])
            print("Start idx: %d, End idx: %d" % (start_idx, end_idx))
            doc_length_norm = self.k * (1 - self.b + self.b * (self.doc_lengths[start_idx:end_idx] / self.avg_doc_len))
            
            term_scores = tfidf_matrix * query_vector
            term_scores_norm = (term_scores * (self.k + 1)) / (tf_matrix + doc_length_norm[:, None])
            
            scores_batch = term_scores_norm.sum(axis=1).flatten()
            scores[start_idx:end_idx] = scores_batch
            
        return scores
    
    def get_top_k(self, scores, top_k):
        """
        Returns the top k documents based on the given scores.

        Args:
            scores (List[float]): The scores of the documents.
            top_k (int): The number of top documents to return.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing the document IDs and their scores,
                                     sorted in descending order of scores.

        """
        return sorted(
            [(self.doc_ids[idx], score) for idx, score in enumerate(scores)],
            key=lambda x: x[1], reverse=True
        )[:top_k]
    