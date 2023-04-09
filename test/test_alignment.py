import numpy as np
import unittest
from src.utils import align_sequences_nw

class AlignSequencesNWTests(unittest.TestCase):
    def test_exact_match(self):
        x = 'AGTACG'
        y = 'AGTACG'
        expected = ['AGTACG', 'AGTACG']
        self.assertEqual(align_sequences_nw(x, y), expected)

    def test_gap_penalty(self):
        x = 'AGTACG'
        y = 'AGC'
        expected = ['AGTACG', 'AG--C-']
        self.assertEqual(align_sequences_nw(x, y), expected)

    def test_custom_parameters(self):
        x = 'AGTACG'
        y = 'AGC'
        match = 2
        mismatch = 1
        gap = 0.5
        expected = ['AGTACG', 'AG--C-']
        self.assertEqual(align_sequences_nw(x, y, match, mismatch, gap), expected)

    def test_empty_sequences(self):
        x = ''
        y = ''
        expected = ['', '']
        self.assertEqual(align_sequences_nw(x, y), expected)
    
    def test_large_gap_penalty(self):
        x = 'AGTACG'
        y = 'AGC'
        gap = 100
        expected = ['AGTACG', 'AG--C-']
        self.assertEqual(align_sequences_nw(x, y, gap=gap), expected)
        

if __name__ == '__main__':
    unittest.main()