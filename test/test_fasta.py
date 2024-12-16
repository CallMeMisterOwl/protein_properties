import unittest
import tempfile
import os
from copy import deepcopy

from src.data.fasta import Fasta


class TestFasta(unittest.TestCase):

    def setUp(self):
        self.fasta_dict = {
            "header1": ["ATCG", "CGAT"],
            "header2": ["GCTA", "TCGA"],
            "header3": ["ATG", "CAT"]
        }
        self.fasta_path = os.path.join(tempfile.gettempdir(), "test.fasta")
        Fasta(sequences=self.fasta_dict).write_fasta(path=self.fasta_path, overwrite=True)

    def tearDown(self):
        if os.path.exists(self.fasta_path):
            os.remove(self.fasta_path)

    def test_read_fasta(self):
        fasta = Fasta(sequences=self.fasta_dict)
        self.assertEqual(fasta.sequences, self.fasta_dict)

        fasta = Fasta(path=self.fasta_path)
        self.assertEqual(fasta.sequences, self.fasta_dict)

    def test_write_fasta(self):
        fasta = Fasta(sequences=self.fasta_dict)
        fasta.write_fasta(path=self.fasta_path, overwrite=True)
        with open(self.fasta_path) as f:
            lines = f.readlines()
        expected_lines = [
            ">header1\n",
            "ATCG\n",
            "CGAT\n",
            ">header2\n",
            "GCTA\n",
            "TCGA\n",
            ">header3\n",
            "ATG\n",
            "CAT\n"
        ]
        self.assertEqual(lines, expected_lines)

        # test file exists error
        with self.assertRaises(FileExistsError):
            fasta.write_fasta(path=self.fasta_path)

    def test_write_fasta_with_nones(self):
        fasta = Fasta(sequences={'header1': [None, None]})
        fasta.write_fasta(path=self.fasta_path, overwrite=True)
        with open(self.fasta_path) as f:
            lines = f.readlines()
        expected_lines = [
            ">header1\n",
            "\n"
        ]
        self.assertEqual(lines, expected_lines)


    def test_get_sequence(self):
        fasta = Fasta(sequences=self.fasta_dict)
        self.assertEqual(fasta.get_sequence("header1"), ["ATCG", "CGAT"])
        self.assertIsNone(fasta.get_sequence("header4"))

    def test_get_sequences(self):
        fasta = Fasta(sequences=self.fasta_dict)
        self.assertEqual(fasta.get_sequences(["header1", "header2"]), [["ATCG", "CGAT"], ["GCTA", "TCGA"]])
        self.assertEqual(fasta.get_sequences(["header1", "header4"]), [["ATCG", "CGAT"], None])

    def test_get_headers(self):
        fasta = Fasta(sequences=self.fasta_dict)
        self.assertEqual(fasta.get_headers(), ["header1", "header2", "header3"])

    def test_get_number_of_sequences_per_header(self):
        fasta = Fasta(sequences=self.fasta_dict)
        self.assertEqual(fasta.get_number_of_sequences_per_header(), 2)

    def test_get_sequence_all_lengths(self):
        fasta = Fasta(sequences=self.fasta_dict)
        assert fasta.get_sequence_all_lengths() == {
            "header1": [4, 4],
            "header2": [4, 4],
            "header3": [3, 3]
        }

    def test_get_sequence_lengths(self):
        fasta = Fasta(sequences=self.fasta_dict)
        # Test single header
        assert fasta.get_sequence_lengths("header1") == {"header1": [4, 4]}
        assert fasta.get_sequence_lengths("header2") == {"header2": [4, 4]}
        assert fasta.get_sequence_lengths("header3") == {"header3": [3, 3]}

        # Test multiple headers
        assert fasta.get_sequence_lengths(["header1", "header2"]) == {"header1": [4, 4], "header2": [4, 4]}
        assert fasta.get_sequence_lengths(["header2", "header3"]) == {"header2": [4, 4], "header3": [3, 3]}

        # Test header not in dictionary
        with self.assertRaises(KeyError):
            fasta.get_sequence_lengths("header4")


    def test_append(self):
        # Test appending Fasta object
        fasta1 = Fasta(sequences={'header1': ['ATCG', 'GTAC']})
        fasta2 = Fasta(sequences={'header1': ['CGAT', 'CATG']})
        fasta1.append(fasta2)
        self.assertEqual(fasta1.get_sequence('header1'), ['ATCG', 'GTAC', 'CGAT', 'CATG'])

        # Test appending dictionary
        fasta1.append({'header1': [[1,2,1,3,4]]})
        self.assertEqual(fasta1.get_sequence('header1'),
                         ['ATCG', 'GTAC', 'CGAT', 'CATG', [1,2,1,3,4]])

        # Test appending dictionary with lists of ints
        fasta3 = Fasta(sequences={'header1': [[1, 2, 3], [4, 5, 6]]})
        fasta3.append({'header1': [[7, 8, 9], [10, 11, 12]]})
        self.assertEqual(fasta3.get_sequence('header1'), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        # Test appending dictionary with lists of floats
        fasta4 = Fasta(sequences={'header1': [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]})
        fasta4.append({'header1': [[7.7, 8.8, 9.9], [10.10, 11.11, 12.12]]})
        self.assertEqual(fasta4.get_sequence('header1'), [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9], [10.10, 11.11, 12.12]])

        # Test appending dictionary with lists of strings
        fasta5 = Fasta(sequences={'header1': [['a', 'b', 'c'], ['d', 'e', 'f']]})
        fasta5.append({'header1': [['g', 'h', 'i'], ['j', 'k', 'l']]})
        self.assertEqual(fasta5.get_sequence('header1'), [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']])

        # Test appending dictionary with lists of mixed types
        fasta6 = Fasta(sequences={'header1': [['a', 1, 1.1], ['d', 2, 2.2]]})
        fasta6.append({'header1': [['g', 3, 3.3], ['j', 4, 4.4]]})
        self.assertEqual(fasta6.get_sequence('header1'), [['a', 1, 1.1], ['d', 2, 2.2], ['g', 3, 3.3], ['j', 4, 4.4]])



    def test_copy(self):
        # Test shallow copy
        fasta1 = Fasta(sequences={'header1': [['ATCG'], ['GTAC']]})
        fasta2 = fasta1.__copy__()
        fasta2.sequences['header1'][0][0] = 'CGAT'
        self.assertEqual(fasta1.get_sequence('header1'), [['CGAT'], ['GTAC']])
        self.assertEqual(fasta2.get_sequence('header1'), [['CGAT'], ['GTAC']])

        # Test deep copy
        fasta3 = deepcopy(fasta1)
        fasta3.sequences['header1'][0][0] = 'TACG'
        self.assertEqual(fasta1.get_sequence('header1'), [['CGAT'], ['GTAC']])
        self.assertEqual(fasta3.get_sequence('header1'), [['TACG'], ['GTAC']])

        


if __name__ == '__main__':
    unittest.main()
