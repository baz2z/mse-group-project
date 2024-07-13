import re
import sys
import tqdm

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from transformers import BertTokenizer, BertModel
from transformers import DebertaV2Tokenizer, DebertaV2Model
import torch
import torch.nn as nn
from in_out import load_corpus_from_json_files

sys.path.insert(0, Path(__file__).resolve().parents[1])


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
        for doc in tqdm.tqdm(self.corpus.values(), desc="Tokenize documents"):
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

class BertEmbedding():
    
    def __init__(self, corpus_path=None, index_path=None):
        """
        A class to create text embeddings.
        
        Args:
            corpus (dict): A dictionary where the keys are document IDs and the values are the
                corresponding documents.
        """
        self.corpus_path = corpus_path       
        self.index_path = index_path
        self.stopwords = set(stopwords.words('english'))
        
        # deberta
        # self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-small')
        # self.model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-small')
        
        # Tiny_BERT embeddings   
        self.tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
        self.model = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
        
        
       
    @property
    def doc_ids(self):
        """
        Ordered document IDs in the corpus.
        """
        return list(self.corpus.keys())

    def get_single_bert_embedding(self, text, Nq=16):
        """
        Generates BERT embeddings for a single text input.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            list: The BERT embeddings for the input text.
        """

        inputs = self.tokenizer(self.remove_stopwords(text), return_tensors="pt", padding=True, truncation=True)

        # TODO: Clean up (still needed?)
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

        # TODO: Clean up
        # # Apply the linear layer to reduce dimension size before normalization
        # reduced_dimension_embedding = self.dimension_reducer(last_hidden_states)

        # # Normalize the reduced embeddings
        # norm = torch.norm(reduced_dimension_embedding, p=2, dim=2, keepdim=True)
        # normalized_reduced_embedding = reduced_dimension_embedding / norm

        # remove embeddings of special beginning and ending token:
        word_embeddings = last_hidden_states[0, 1:-1, :]

        
        return word_embeddings

    # refactor idea: write bert embeddings to numpy array here instead of to one big dict
    def create_bert_embeddings(self):
        """
        Generates BART embeddings for each term in the corpus, handling documents longer than the maximum sequence length by chunking.
        
        Returns:
            A dictionary where keys are document IDs and values are concatenated embeddings of chunks.
        """
        self.initiate_corpus()
        doc_ids = []  # List to store document IDs
        chunk_embeddings_list = []  # List to temporarily store embeddings for each document
        
        max_length = 512  # Assuming 512 is the max length for BART


        # Step 1: Calculate the maximum number of tokens across all documents
        max_tokens = 0
        for doc_id, document in self.corpus.items():
            document = self.remove_stopwords(document)
            if not document:
                document = " "
            inputs = self.tokenizer(document, return_tensors="pt", padding=True, truncation=True)
            num_tokens = inputs.input_ids.size(1)
            if num_tokens > max_tokens:
                max_tokens = num_tokens

        
        # Step 2: Process each document, pad if necessary, and save
        for doc_id, document in self.corpus.items():
            document = self.remove_stopwords(document)
            if not document:
                document = " "
            chunks = [document[i:i+max_length] for i in range(0, len(document), max_length)]
            chunk_embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                chunk_embeddings.append(last_hidden_states)
            concatenated_embeddings = torch.cat(chunk_embeddings, dim=1)[0]

            # Padding
            padding_length = max_tokens - concatenated_embeddings.shape[0]
            if padding_length > 0:
                padded_embedding = torch.nn.functional.pad(concatenated_embeddings, (0, 0, 0, padding_length), "constant", 0)
            else:
                padded_embedding = concatenated_embeddings

            # Save the padded_embedding tensor to a file named after the doc_id
            tensor_filename = f"{self.index_path}/{doc_id}.pt"  # Assuming self.index_path is the directory where you want to save
            torch.save(padded_embedding, tensor_filename)

    
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

    def initiate_corpus(self):
        """
        Load the corpus from the given path.
        """
        self.corpus = load_corpus_from_json_files(self.corpus_path)

    def save_tensor(tensor, file_path, compress=False):
        if compress:
            with gzip.open(file_path + '.gz', 'wb') as f:
                torch.save(tensor, f)
        else:
            torch.save(tensor, file_path)
