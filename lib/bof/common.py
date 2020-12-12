import gzip
import errno
import os
import dill as pickle

from pathlib import Path


class MixInIO:
    """
    Provide basic save/load capacities to other classes.
    """

    def save(self, filename: str, path='.', erase=False, compress=False):
        """
        Save instance to file.

        Parameters
        ----------
        filename: str
            The stem of the filename.
        path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
            The location path.
        erase: bool
            Should existing file be erased if it exists?
        compress: bool
            Should gzip compression be used?

        Examples
        ----------
        >>> import tempfile
        >>> from bof.factortree import FactorTree
        >>> tree1 = FactorTree(["riri", "fifi", "rififi"])
        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     tree1.save(filename='myfile', compress=True, path=tmpdirname)
        ...     dir_content = [f.name for f in Path(tmpdirname).glob('*')]
        ...     tree2 = FactorTree(filename='myfile', path=Path(tmpdirname))
        ...     tree1.save(filename='myfile', compress=True, path=tmpdirname) # doctest.ELLIPSIS
        File ...myfile.pkl.gz already exists! Use erase option to overwrite.
        >>> dir_content
        ['myfile.pkl.gz']
        >>> tree2.self_factors
        [8, 8, 15]

        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     tree1.save(filename='myfile', path=tmpdirname)
        ...     tree1.save(filename='myfile', path=tmpdirname) # doctest.ELLIPSIS
        File ...myfile.pkl already exists! Use erase option to overwrite.

        >>> tree1.add_txt_to_tree("titi")
        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     tree1.save(filename='myfile', path=tmpdirname)
        ...     tree1.save(filename='myfile', path=tmpdirname, erase=True)
        ...     tree2.load(filename='myfile', path=tmpdirname)
        ...     dir_content = [f.name for f in Path(tmpdirname).glob('*')]
        >>> dir_content
        ['myfile.pkl']
        >>> tree2.corpus_list
        ['riri', 'fifi', 'rififi', 'titi']

        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...    tree2.load(filename='thisfilenamedoesnotexist') # doctest.ELLIPSIS
        Traceback (most recent call last):
         ...
        FileNotFoundError: [Errno 2] No such file or directory: ...
        """
        path = Path(path)
        destination = path / Path(filename).stem
        if compress:
            destination = destination.with_suffix(".pkl.gz")
            if destination.exists() and not erase:
                print(f"File {destination} already exists! Use erase option to overwrite.")
            else:
                with gzip.open(destination, "wb") as f:
                    pickle.dump(self, f)
        else:
            destination = destination.with_suffix(".pkl")
            if destination.exists() and not erase:
                print(f"File {destination} already exists! Use erase option to overwrite.")
            else:
                with open(destination, "wb") as f:
                    pickle.dump(self, f)

    def load(self, filename: str, path='.'):
        """
        Load instance from file.

        Parameters
        ----------
        filename: str
            The stem of the filename.
        path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
            The location path.
        """
        path = Path(path)
        dest = path / Path(filename).with_suffix(".pkl")
        if dest.exists():
            with open(dest, 'rb') as f:
                self.__dict__.update(pickle.load(f).__dict__)
        else:
            dest = dest.with_suffix('.pkl.gz')
            if dest.exists():
                with gzip.open(dest) as f:
                    self.__dict__.update(pickle.load(f).__dict__)
            else:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), dest)


def default_preprocessor(txt):
    """
    Default string preprocessor: trim extra spaces and lower case from string `txt`.

    Parameters
    ----------
    txt: :py:class:`str`
        Text to process.

    Returns
    -------
    :py:class:`str`
        Processed text.

    Examples
    ---------
    >>> default_preprocessor(" LaTeX RuleZ    ")
    'latex rulez'
    """
    return txt.strip().lower()
