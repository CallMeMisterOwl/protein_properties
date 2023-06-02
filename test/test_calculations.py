import unittest
import os
from src.data.fasta import Fasta
from src.data.calculate_SASA_Bfactor_flip import calculate_scores_for_protein, calculate_scores
from biotite.structure.io.pdbx import PDBxFile, get_structure
from biotite.structure import sasa


class TestCalculateScores(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary directory for testing
        cls.test_dir = os.path.join(os.getcwd(), "test_data")
        os.makedirs(cls.test_dir, exist_ok=True)
        # Create temporary FASTA file for testing
        cls.fasta_file = os.path.join(cls.test_dir, "test.fasta")
        with open(cls.fasta_file, "w") as f:
            f.write(">1ls1-A\n")
            f.write("MENKVLIEGELANVSVLGGAKKLKDLAEEIATYYK\n")
            f.write(">12as-A\n")
            f.write("TETSQVAPA\n")
        # Set path to PDB structures
        cls.pdb_path = os.path.join(cls.test_dir, "pdb_structures")
        os.makedirs(cls.pdb_path, exist_ok=True)
        # Copy CIF files for testing
        os.system(f"cp data/1ls1.cif {cls.pdb_path}")
        os.system(f"cp data/12as.cif {cls.pdb_path}")

    def test_calculate_scores_for_protein(self):
        # Test with one protein
        protein = "1ls1-A"
        pdb_path = self.pdb_path
        sasa_scores, bfactor_scores = calculate_scores_for_protein(protein, pdb_path)
        self.assertIsInstance(sasa_scores, list)
        self.assertIsInstance(bfactor_scores, list)
        self.assertEqual(len(sasa_scores), 202)
        self.assertEqual(len(bfactor_scores), 202)
        # Test with non-existing protein
        protein = "2abc-A"
        pdb_path = self.pdb_path
        with self.assertRaises(FileNotFoundError):
            sasa_scores, bfactor_scores = calculate_scores_for_protein(protein, pdb_path)

    def test_calculate_scores(self):
        # Test with one protein
        fasta_file = Fasta(self.fasta_file)
        pdb_path = self.pdb_path
        sasa_scores, bfactor_scores = calculate_scores(fasta_file, pdb_path)
        self.assertIsInstance(sasa_scores, dict)
        self.assertIsInstance(bfactor_scores, dict)
        self.assertEqual(len(sasa_scores), 2)
        self.assertEqual(len(bfactor_scores), 2)
        self.assertEqual(len(sasa_scores["12as-A"]), 5385)
        self.assertEqual(len(bfactor_scores["12as-A"]), 5385)
        # Test with non-existing protein
        fasta_file = Fasta(self.fasta_file)
        fasta_file._sequences["2abc-A"] = ["MENKVLIEGELANVSVLGGAKKLKDLAEEIATYYK"]
        pdb_path = self.pdb_path
        with self.assertRaises(FileNotFoundError):
            sasa_scores, bfactor_scores = calculate_scores(fasta_file, pdb_path)



    @classmethod
    def tearDownClass(cls):
        # Delete temporary directory and files
        os.system(f"rm -rf {cls.test_dir}")


if __name__ == "__main__":
    unittest.main()
