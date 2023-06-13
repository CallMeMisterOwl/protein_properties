import unittest
import os

import numpy as np
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
        cls.pdb_path = "test/data/"
        os.makedirs(cls.pdb_path, exist_ok=True)
        # Copy CIF files for testing

    def test_calculate_scores_for_protein(self):
        protein1 = "1ls1-A"
        pdb_path1 = self.pdb_path
        map_missing_res1 = ["---------------------------------------------------------------------------",
        "---------------------------------------------------------------------------",
        "---------------------------------------------------------------------------",
        "----------------------------------------------XXXXXX------------------"]
        expected_output1 = ("1ls1-A", np.array([0.0, 0.0, ...]), np.array([0.0, 0.0, ...]))

        protein2 = "12as-A"
        pdb_path2 = self.pdb_path
        map_missing_res2 = ["XXX------------------------------------------------------------------------", "---------------------------------------------------------------------------", "---------------------------------------------------------------------------", "---------------------------------------------------------------------------", "------------------------------"]
        expected_output2 = ("12as-A", np.array([0.0, 0.0, ...]), np.array([0.0, 0.0, ...]))

        # Add more test cases if needed

        calculate_scores_for_protein(protein1, pdb_path1, map_missing_res1) == expected_output1
        calculate_scores_for_protein(protein2, pdb_path2, map_missing_res2) == expected_output2



    @classmethod
    def tearDownClass(cls):
        # Delete temporary directory and files
        os.system(f"rm -rf {cls.test_dir}")


if __name__ == "__main__":
    unittest.main()
