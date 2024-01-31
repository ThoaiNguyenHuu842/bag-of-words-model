from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from typing import List

nltk.download('punkt')
nltk.download('stopwords')


def bow(documents: List[str]):
    bow_vectors : List = []
    vocabulary = set()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    for document in documents:
        # Tokenization
        document = word_tokenize(document)
        # Stopwords removal
        document = [word for word in document if word.lower() not in stop_words]
        # Stemming
        document = [stemmer.stem(word) for word in document]
        # Create vocabulary
        vocabulary.update(document)

    # sort vocabulary alphabetical to make to result predictable
    vocabulary = sorted(list(vocabulary))
    for document in documents:
        # Bag-of-Words representation
        bow_vectors.append([document.count(word) for word in vocabulary])
    return {"bow_vectors": bow_vectors, "vocabulary": vocabulary}


# Sample documents
document1 = "Natural language processing is an exciting field of study"
document2 = "Studying NLP involves understanding various language processing techniques such as language detection"

print(bow([document1, document2]))