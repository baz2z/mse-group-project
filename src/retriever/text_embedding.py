import re
import sys

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from transformers import BertTokenizer, BertModel
import torch

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
        
        # Tiny_BERT embeddings 
        self.tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
        self.model = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
    
    def get_doc_ids(self):
        print(self.corpus.keys())
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


    def get_bert_embeddings(self):
        """
        Generates BART embeddings for each term in the corpus, handling documents longer than the maximum sequence length by chunking.
        
        Returns:
            A dictionary where keys are document IDs and values are concatenated embeddings of chunks.
        """
        embeddings = {}
        max_length = 512  # Assuming 512 is the max length for BART
        for doc_id, document in self.corpus.items():
            # Ensure the document is not empty by appending a space if it is
            if not document:
                document = " "
            # Split document into chunks
            chunks = [document[i:i+max_length] for i in range(0, len(document), max_length)]
            chunk_embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                chunk_embeddings.append(last_hidden_states)
            
            # Concatenate embeddings from all chunks
            concatenated_embeddings = torch.cat(chunk_embeddings, dim=1)
            embeddings[doc_id] = concatenated_embeddings.tolist()
        return embeddings

    
    
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
