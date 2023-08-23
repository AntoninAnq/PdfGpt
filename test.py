import unittest
import utils
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Test(unittest.TestCase):

    def test_test_cpu_arch(self):
        """
        Test the CPU check for Apple M1 or Intel chip
        """
        result = utils.test_cpu_arch()
        cpu_list = ['arm', 'x86_64']
        self.assertIn(result, cpu_list)


    def test_setup_QA(self):
        """Integration test for testing the whole chain"""
        path_to_model = '/Users/antonin/code/PersonalGpt/models/llama-2-13b-chat.ggmlv3.q4_0.bin'
        path_to_db = '/Users/antonin/code/PersonalGpt/test/test_vector_db/db_faiss'
        dbqa = utils.setup_QA(path_to_db,path_to_model)
        prompt = "What is a window function ?"
        response = dbqa({'query':prompt} )
        print(f"Response : {response['result']}")
        sentence = "A window function is a function that operates on multiple \
            rows or groups of rows within the result set returned from a query,\
                similar to standard aggregate functions, but the groups of rows\
                    are defined not by a GROUP BY clause, but by partitioning and windowing clauses."
        "Tokenize response and testing sentence"
        token_response = word_tokenize(response['result'])
        token_test_sentence = word_tokenize(sentence)

        "Removing the stop word"
        sw = stopwords.words('english')
        set_response = {i for i in token_response if not i in sw}
        set_test_sentence = {i for i in token_test_sentence if not i in sw}

        "Create vector for each string linked with a set of the two strings"
        vector = set_response.union(set_test_sentence)
        response_list = list()
        sentence_list = list()

        for i in vector:
            if i in set_response:
                response_list.append(1)
            else:
                response_list.append(0)
            if i in set_test_sentence:
                sentence_list.append(1)
            else:
                sentence_list.append(0)
        c=0

        "Compute consine similarity"
        for i in range(len(vector)):
            c+=response_list[i]*sentence_list[i]
        cosine = c / float((sum(response_list)*sum(sentence_list))**0.5)
        print(f"Similarity : {cosine}")
        self.assertGreater(cosine,0.5)

