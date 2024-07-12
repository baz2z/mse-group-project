import os
import json
import csv

from scipy.sparse import csr_matrix


def load_corpus_from_json_files(directory_path, k=200):
    corpus = {}
    idx = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            if idx > k:
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