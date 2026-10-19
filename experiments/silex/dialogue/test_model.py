import unittest
import numpy as np
from model import encode_questions, vocabulary, init, generate, tokens, check_gradients


class ModelTests(unittest.TestCase):
    def test_gradients(self):
        check_gradients()

    def test_question_is_order_blind(self):
        pairs=[('la france est la capitale de paris','paris est la capitale de la france')]
        vocab=vocabulary(pairs)
        bags=encode_questions(list(pairs[0]),vocab)
        np.testing.assert_array_equal(bags[0],bags[1])

    def test_unknown_token(self):
        vocab=vocabulary([('bonjour','salut')])
        bag=encode_questions(['mot_absent'],vocab)
        self.assertEqual(float(bag[0,3]),1.0)

    def test_generation_bounded_deterministic(self):
        vocab=vocabulary([('bonjour','salut')])
        p=init(len(vocab),4)
        a=generate(p,vocab,'bonjour',limit=3)
        b=generate(p,vocab,'bonjour',limit=3)
        self.assertEqual(a,b)
        self.assertLessEqual(a['generated_tokens'],3)
        self.assertNotIn('<pad>',a['answer'])

    def test_empty_question_encoding(self):
        bag=encode_questions([''],vocabulary([('a','b')]))
        self.assertTrue(np.isfinite(bag).all())
        self.assertEqual(float(bag.sum()),0)


if __name__=='__main__':
    unittest.main()
