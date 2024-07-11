from transformers import pipeline


class DebertaV3:
    
    def __init__(self, corpus) -> None:
        
        task = "zero-shot-classification"
        model = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
        self.zeroshot_classifier = pipeline(task, model=model) 
        
        self.doc_ids = list(corpus.keys())
        self.corpus = list(corpus.values())
        self.hypothesis_template = "This text is about {}"
    
    def rank(self, query, top_k=5): 
        """
        Rank the documents based on the given query.

        Args:
            query (str): The query string.
            top_k (int, optional): The number of documents to return. Defaults to 5.

        Returns:
            list: A list of tuples containing the document ID and the relevance score.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a single string")
        
        scores = []
        for document in self.corpus:
            score = self.zeroshot_classifier(
                document, [query], 
                hypothesis_template=self.hypothesis_template, 
                multi_label=False)['scores']
            scores.extend(score)
        
        return self.get_top_k(scores, top_k)

    def get_top_k(self, scores, top_k=5):
        """
        Retrieve the top k documents from the scores.

        Args:
            scores (list): The list of scores.
            top_k (int, optional): The number of documents to return. Defaults to 5.

        Returns:
            list: A list of tuples containing the document ID and the relevance score.
        """
        results = sorted(
            [(self.doc_ids[idx], score) for idx, score in enumerate(scores)],
            key=lambda x: x[1], reverse=True
        )
        
        if top_k > len(results):
            top_k = len(results)
        
        return results[:top_k]
        
        
def main():
    
    # Try on a small corpus of random documents
    corpus = {
        "doc1": "Angela Merkel is a politician in Germany and leader of the CDU",
        "doc2": "The economy is going down the drain",
        "doc3": "The new movie is a hit",
        "doc4": "The environment is in danger"
        }
    
    query = "politics enviroment"
    
    deberta = DebertaV3(corpus)
    top_n = deberta.rank(query)
    
    for doc_id, score in top_n:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")

    
if __name__ == "__main__":
    main()
    