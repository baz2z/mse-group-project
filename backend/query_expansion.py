""" Contains functions to expand a given user query """
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

def expand_query_with_wordnet(query, category):
    """
    Expand a query by adding synonyms from WordNet and including a specified category.

    This function takes a query string, splits it into individual words, and adds synonyms
    for each word using WordNet. It also includes the specified category in the expansion.

    :param query: The query string to expand.
    :param category: The category to include in the expanded query.
    :return: The expanded query string with additional synonyms and the category.
    """
    words = query.split()
    expanded_words = set(words)
    
    # Add category to the set of words to expand
    words.append(category)

    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded_words.add(lemma.name())
                
    return ' '.join(expanded_words)

