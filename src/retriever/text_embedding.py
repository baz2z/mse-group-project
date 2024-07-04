import re
import sys

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

sys.path.insert(0, Path(__file__).resolve().parents[1])


class TextEmbedding():
    
    def __init__(self, corpus=None):
        """
        A class to create text embeddings.
        
        Args:
            corpus (dict): A dictionary where the keys are document IDs and the values are the
                corresponding documents (strings):
                
                Example:
                    {
                        'doc1': 'This is the first document.',
                        'doc2': 'This is the second document.'
                    }

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
