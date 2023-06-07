from pathlib import Path
from typing import Optional, Union

# TODO implement functionality to recognize std fasta files and automatically concate the lines 
class Fasta:
    """
    Class to read and write fasta files and store the sequences in a dictionary.
    The keys of the dictionary are the headers of the sequences and the values are the sequences.

    Attributes
    ----------
    _sequences : dict
        Dictionary with the sequences. The keys are the headers and the values are the sequences.

    """

    def __init__(self, path: Optional[str] = None, sequences: Optional[dict] = None):
        self._sequences = {} if sequences is None else sequences
        if path is not None:
            self.read_fasta(path)

    def read_fasta(self, path: str):
        """
        Reads a fasta file and stores the sequences in a dictionary.
        :param path:
        :return: None
        """
        path = Path(path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f'{path} does not exist or is not a file')

        with open(path, 'r') as f:
            lines = f.readlines()
        header = None
        sequence = []
        for idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    self._sequences[header] = sequence
                    sequence = []
                if not line[1:]:
                    raise ValueError(f'Fasta file {path} contains an empty header at line {idx}')
                header = line[1:]
            elif "|" in line:
                iorf = float if "." in line else int
                sequence.append([iorf(d) for d in line.split('|')])
            else:
                sequence.append(line)
        self._sequences[header] = sequence

    def write_fasta(self, path: str, overwrite: bool = False):
        """
        Writes the sequences in a fasta file.
        :param overwrite: bool Overwrite the file if it exists
        :param path: str Path to the fasta file
        :return: None
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f'File {path} already exists')
        path.parents[0].mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for header, sequence in self._sequences.items():
                f.write(f'>{header}\n')
                f.write('\n'.join([i if isinstance(i, str) else '|'.join([str(x) for x in i]) for i in sequence]) + '\n')

    def get_sequence(self, header: str) -> Optional[list]:
        """
        Returns the sequence of a given header.
        :param header:
        :return: Optional[str]
        """
        if header in self._sequences:
            return self._sequences[header]
        else:
            return None

    def get_sequences(self, headers: list) -> Optional[list]:
        """
        Returns the sequences of a given list of headers.
        :param headers: list
        :return: Optional[list]
        """
        sequences = []
        for header in headers:
            if header in self._sequences:
                sequences += [self._sequences[header]]
            else:
                sequences += [None]
        return sequences

    def get_headers(self) -> list:
        """
        Returns the headers of the sequences.
        :return: list
        """
        return list(self._sequences.keys())

    def get_number_of_sequences_per_header(self) -> int:
        """
        Returns the number of sequences per header i.e. how many lines per header exist.
        :return: int
        """
        return len(list(self._sequences.values())[0])

    # combine the function names to get_sequence_lengths into one function

    def get_sequence_lengths(self, headers: Union[str, list]) -> Optional[dict]:
        """
        Returns the lengths of the sequences of a given list of headers. If the header is not in the dictionary, None is returned.
        :param headers: Union[str, list]
        :return: Optional[dict]
        """
        if isinstance(headers, str):
            headers = [headers]

        lengths = {}
        for header in headers:
            if header in self._sequences:
                lengths[header] = [len(i) for i in self._sequences[header]]
            else:
                raise KeyError(f'Header {header} not in dictionary')
        return lengths

    def get_sequence_all_lengths(self) -> dict:
        """
        Returns the lengths of the sequences.
        :return:
        """
        return {header: [len(i) for i in sequence] for header, sequence in self._sequences.items()}

    def append(self, other: Union['Fasta', dict]):
        """
        Appends the sequences of another Fasta object or a dictionary to the current Fasta object.
        Note the function expects the values to be lists.
        :param other:
        :return:
        """
        if isinstance(other, Fasta):
            for key, value in other._sequences.items():
                if isinstance(value, list):
                    self._sequences.setdefault(key, []).extend(value)
                elif isinstance(value, str):
                    self._sequences.setdefault(key, []).append(value)
        elif isinstance(other, dict):
            for key, value in other.items():
                if isinstance(value, list):
                    self._sequences.setdefault(key, []).extend(value)
                elif isinstance(value, str):
                    self._sequences.setdefault(key, []).append(value)
        return self

    def __copy__(self):
        return Fasta(sequences=self._sequences)

    def __repr__(self):
        return f'Fasta file with {len(self._sequences)} sequences'

    def __str__(self):
        return f'Fasta file with {len(self._sequences)} sequences'

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, key):
        return self._sequences[key]

    def __setitem__(self, key, value):
        self._sequences[key] = value

    def __delitem__(self, key):
        del self._sequences[key]

    def __iter__(self):
        return iter(self._sequences)

    def __contains__(self, key):
        return key in self._sequences

    def __add__(self, other):
        if isinstance(other, Fasta):
            self._sequences.update(other._sequences)
        else:
            raise TypeError(f'Cannot add Fasta and {type(other)}')
        return self

    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other
