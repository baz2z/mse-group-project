import re
import sys

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

sys.path.insert(0, Path(__file__).resolve().parents[1])


class TextEmbedding():
    
    def __init__(self, corpus=None, output_dim=8):
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
        self.dimension_reducer = nn.Linear(self.model.config.hidden_size, output_dim)

    
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


    def get_single_bert_embedding(self, text, Nq=16):
        """
        Generates BERT embeddings for a single text input.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            list: The BERT embeddings for the input text.
        """

        inputs = self.tokenizer(self.remove_stopwords(text), return_tensors="pt", padding=True, truncation=True)

        # # Get the actual length of input_ids
        # input_length = inputs['input_ids'].size(1)

        # # Pad or truncate
        # if input_length < Nq:
        #     # Calculate the number of mask tokens to add
        #     num_mask_tokens = Nq - input_length
        #     # Create a tensor of mask tokens
        #     mask_tokens = torch.full((1, num_mask_tokens), self.tokenizer.mask_token_id, dtype=torch.long)
        #     # Concatenate the mask tokens to the input_ids
        #     inputs['input_ids'] = torch.cat([inputs['input_ids'], mask_tokens], dim=1)
        #     # Also, adjust the attention_mask to account for the added mask tokens
        #     attention_mask = torch.cat([inputs['attention_mask'], torch.ones((1, num_mask_tokens), dtype=torch.long)], dim=1)
        #     inputs['attention_mask'] = attention_mask
        # elif input_length > Nq:
        #     # Truncate the input_ids and attention_mask to Nq tokens
        #     inputs['input_ids'] = inputs['input_ids'][:, :Nq]
        #     inputs['attention_mask'] = inputs['attention_mask'][:, :Nq]

        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        # Apply the linear layer to reduce dimension size before normalization
        reduced_dimension_embedding = self.dimension_reducer(last_hidden_states)

        # Normalize the reduced embeddings
        norm = torch.norm(reduced_dimension_embedding, p=2, dim=2, keepdim=True)
        normalized_reduced_embedding = reduced_dimension_embedding / norm
        return normalized_reduced_embedding.tolist()


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
            document = self.remove_stopwords(document)

            if not document:
                document = " "
            # remove stopwords of document
            # Split document into chunks
            chunks = [document[i:i+max_length] for i in range(0, len(document), max_length)]
            chunk_embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                # Apply the linear layer to reduce dimension size
                reduced_dimension_embedding = self.dimension_reducer(last_hidden_states)
                chunk_embeddings.append(reduced_dimension_embedding)
            
            # Concatenate embeddings from all chunks
            concatenated_embeddings = torch.cat(chunk_embeddings, dim=1)
            # normaliize concatenated embeddings

            # Calculate the Euclidean norm (magnitude) of the vector
            magnitude = torch.norm(concatenated_embeddings, p=2, dim=2, keepdim=True)
            # Normalize the vector by dividing by its magnitude
            normalized_embeddings = concatenated_embeddings / magnitude

            embeddings[doc_id] = normalized_embeddings.tolist()
        return embeddings
    
    def remove_stopwords(self, text):
        """
        Removes stopwords from the given text.

        Args:
            text (str): The text from which to remove stopwords.

        Returns:
            str: The text with stopwords removed.
        """
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stopwords]
        return ' '.join(tokens)

    
    
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
