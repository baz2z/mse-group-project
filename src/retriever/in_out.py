import os
import json
import csv

from scipy.sparse import csr_matrix


def load_corpus(index_dir, k=None):
    """
    Load a corpus from a directory of md files.

    Args:
        dir (str): Path to folder containing md files.
        k (int): Number of documents to load.
    """
    corpus = {}
    idx = 0
    for filename in os.listdir(index_dir):
        if k is not None and idx >= k:
            return corpus
        file_path = os.path.join(index_dir, filename)
        with open(file_path, 'r') as file:
            corpus[filename[:-8]] = file.read()
        idx += 1
        if idx % 5000 == 0:
            print(f"Processing files - n={idx}")
    print(f"Loaded {idx} documents.")
    return corpus


def create_subset_folder(index_dir, subset_dir, k=200):
    """
    Create a subset of a corpus from a directory of md files.

    Args:
        index_dir (str): Path to folder containing md files.
        subset_dir (str): Path to folder where subset will be saved.
        k (int): Number of documents to save.
    """
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
    idx = 0
    for filename in os.listdir(index_dir):
        if idx >= k:
            return
        file_path = os.path.join(index_dir, filename)
        with open(file_path, 'r') as file:
            text = file.read()
        with open(os.path.join(subset_dir, filename), 'w') as file:
            file.write(text)
        idx += 1
    print(f"saved {idx} documents")


def load_corpus_from_json_files(directory_path, k=100000):
    corpus = {}
    idx = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            if idx >= k:
                return corpus
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                corpus[filename[:-5]] = data['text']
            idx += 1
        if idx % 5000 == 0:
            print(f"Processing jsons - n={idx}")
    print(f"loaded {idx} documents")
    return corpus


def load_url_from_json_files(directory_path, k=1000):
    urls = {}
    idx = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            if idx > k:
                return urls
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                urls[filename[:-5]] = data['url']
            idx += 1
        if idx % 5000 == 0:
            print(f"Processing jsons - n={idx}")
    return urls


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


def convert_csr_to_dict(obj):
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


def load_csr_matrix(data):
    if data['type'] == 'csr_matrix':
        return csr_matrix(
            (data['data'], data['indices'], data['indptr']), 
            shape=data['shape']
        )
    else:
        raise ValueError("JSON does not contain csr_matrix data")
    

if __name__ == "__main__":
    
    corpus = load_corpus("dat/eng", 4000)
    url_map = load_url_mapping_from_csv("dat/eng_doc_id_mapping.csv")
    
    print(f"Loaded {len(corpus)} documents")
    print(list(corpus.items())[:2])
    
    create_subset_folder("dat/eng", "dat/eng_subset", 200)