import unittest
from bow_app import bow

class TestStringMethods(unittest.TestCase):

    def test_bag_of_words_two_docs(self):
        document1 = "Natural language processing is an exciting field of study"
        document2 = "Studying NLP involves understanding various language processing techniques such as language detection"

        result = bow([document1, document2])

        self.assertEqual(result['vocabulary'], ['detect', 'excit', 'field', 'involv', 'languag', 'natur', 'nlp', 'process', 'studi', 'techniqu', 'understand', 'variou'])
        self.assertEqual(result['bow_vectors'],[[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 2, 0, 0, 1, 0, 1, 1, 1]])

    def test_bag_of_words_basic(self):
        document1 = "This is a test document."

        result = bow([document1])

        self.assertEqual(result['vocabulary'], ['.', 'document', 'test'])
        self.assertEqual(result['bow_vectors'],[[1, 1, 1]])

    def test_bag_of_words_empty_document(self):
        document1 = ""

        result = bow([document1])

        self.assertEqual(result['vocabulary'], [])
        self.assertEqual(result['bow_vectors'],[[]])

    def test_bag_of_words_multiple_occurrences(self):
        document1 = "This is an example. This example is simple."

        result = bow([document1])

        self.assertEqual(result['vocabulary'],['.', 'exampl', 'simpl'])
        self.assertEqual(result['bow_vectors'], [[2, 2, 1]])

if __name__ == '__main__':
    unittest.main()  