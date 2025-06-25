import gzip
import errno
import os
import dill as pickle
from pathlib import Path


class MixInIO:
    """
    Provide basic save/load capacities to other classes.
    """

    def dump(
        self, filename: str, path=".", overwrite=False, compress=True, stemize=True
    ):
        """
        Save instance to file.

        Parameters
        ----------
        filename: str
            The stem of the filename.
        path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
            The location path.
        overwrite: bool, default=False
            Should existing file be erased if it exists?
        compress: bool, default=True
            Should gzip compression be used?
        stemize: bool, default=True
            Trim any extension (e.g. .xxx)

        Examples
        ----------

        >>> import tempfile
        >>> v1 = ToyClass(42)
        >>> v2 = ToyClass()
        >>> v2.value
        0
        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     v1.dump(filename='myfile', compress=True, path=tmpdirname)
        ...     dir_content = [file.name for file in Path(tmpdirname).glob('*')]
        ...     v2 = ToyClass.load(filename='myfile', path=Path(tmpdirname))
        ...     v1.dump(filename='myfile', compress=True, path=tmpdirname) # doctest.ELLIPSIS
        File ...myfile.pkl.gz already exists! Use overwrite option to overwrite.
        >>> dir_content
        ['myfile.pkl.gz']
        >>> v2.value
        42

        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     v1.dump(filename='myfile', compress=False, path=tmpdirname)
        ...     v1.dump(filename='myfile', compress=False, path=tmpdirname) # doctest.ELLIPSIS
        File ...myfile.pkl already exists! Use overwrite option to overwrite.

        >>> v1.value = 51
        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     v1.dump(filename='myfile', path=tmpdirname, compress=False)
        ...     v1.dump(filename='myfile', path=tmpdirname, overwrite=True, compress=False)
        ...     v2 = ToyClass.load(filename='myfile', path=tmpdirname)
        ...     dir_content = [file.name for file in Path(tmpdirname).glob('*')]
        >>> dir_content
        ['myfile.pkl']
        >>> v2.value
        51

        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...    v2 = ToyClass.load(filename='thisfilenamedoesnotexist')
        Traceback (most recent call last):
         ...
        FileNotFoundError: [Errno 2] No such file or directory: ...
        """
        path = Path(path)
        fn = Path(filename)
        if stemize:
            fn = Path(fn.stem)
        if compress:
            destination = path / (fn.name + ".pkl.gz")
            if destination.exists() and not overwrite:
                print(
                    f"File {destination} already exists! Use overwrite option to overwrite."
                )
            else:
                with gzip.open(destination, "wb") as f:
                    pickle.dump(self, f)
        else:
            destination = path / (fn.name + ".pkl")
            if destination.exists() and not overwrite:
                print(
                    f"File {destination} already exists! Use overwrite option to overwrite."
                )
            else:
                with open(destination, "wb") as f:
                    pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str, path="."):
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
            with open(dest, "rb") as f:
                return pickle.load(f)
        else:
            dest = dest.with_suffix(".pkl.gz")
            if dest.exists():
                with gzip.open(dest) as f:
                    return pickle.load(f)
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dest)


class ToyClass(MixInIO):
    def __init__(self, value=0):
        self.value = value


# import gzip
# import errno
# import os
# import dill as pickle
# import numpy as np
# from numba import njit

# from pathlib import Path


# class MixInIO:
#     """
#     Provide basic save/load capacities to other classes.
#     """
#
#     def save(self, filename: str, path='.', erase=False, compress=False):
#         """
#         Save instance to file.
#
#         Parameters
#         ----------
#         filename: str
#             The stem of the filename.
#         path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
#             The location path.
#         erase: bool
#             Should existing file be erased if it exists?
#         compress: bool
#             Should gzip compression be used?
#
#         Examples
#         ----------
#         >>> import tempfile
#         >>> from bof.feature_extraction import CountVectorizer
#         >>> vect1 = CountVectorizer(n_range=3)
#         >>> vect1.fit(["riri", "fifi", "rififi"])
#         >>> with tempfile.TemporaryDirectory() as tmpdirname:
#         ...     vect1.save(filename='myfile', compress=True, path=tmpdirname)
#         ...     vect2 = CountVectorizer(filename='myfile', path=Path(tmpdirname))
#         >>> vect2.features
#         ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
#
#         >>> with tempfile.TemporaryDirectory() as tmpdirname:
#         ...     vect1.save(filename='myfile', compress=True, path=tmpdirname)
#         ...     dir_content = [f.name for f in Path(tmpdirname).glob('*')]
#         ...     vect2 = CountVectorizer(filename='myfile', path=Path(tmpdirname))
#         ...     vect1.save(filename='myfile', compress=True, path=tmpdirname) # doctest.ELLIPSIS
#         File ...myfile.pkl.gz already exists! Use erase option to overwrite.
#         >>> dir_content
#         ['myfile.pkl.gz']
#         >>> vect2.m
#         12
#
#         >>> from bof.fuzz import Process
#         >>> p1 = Process()
#         >>> p1.fit(["riri", "fifi", "rififi"])
#         >>> with tempfile.TemporaryDirectory() as tmpdirname:
#         ...     p1.save(filename='myfile', compress=True, path=tmpdirname)
#         ...     p2 = Process(filename='myfile', path=Path(tmpdirname))
#         >>> p2.extractOne("rififo")
#         ('rififi', 63.1578947368421)
#
#         >>> with tempfile.TemporaryDirectory() as tmpdirname:
#         ...     p1.save(filename='myfile', path=tmpdirname)
#         ...     p1.save(filename='myfile', path=tmpdirname) # doctest.ELLIPSIS
#         File ...myfile.pkl already exists! Use erase option to overwrite.
#
#         >>> vect1.fit_transform(["titi"], reset=False).toarray()
#         array([[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1]], dtype=uint32)
#         >>> with tempfile.TemporaryDirectory() as tmpdirname:
#         ...     vect1.save(filename='myfile', path=tmpdirname)
#         ...     vect1.save(filename='myfile', path=tmpdirname, erase=True)
#         ...     vect2.load(filename='myfile', path=tmpdirname)
#         ...     dir_content = [f.name for f in Path(tmpdirname).glob('*')]
#         >>> dir_content
#         ['myfile.pkl']
#         >>> vect2.features
#         ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif', 't', 'ti', 'tit', 'it', 'iti']
#
#         >>> with tempfile.TemporaryDirectory() as tmpdirname:
#         ...    vect2.load(filename='thisfilenamedoesnotexist', path=tmpdirname) # doctest.ELLIPSIS
#         Traceback (most recent call last):
#          ...
#         FileNotFoundError: [Errno 2] No such file or directory: ...
#         """
#         path = Path(path)
#         destination = path / Path(filename).stem
#         if compress:
#             destination = destination.with_suffix(".pkl.gz")
#             if destination.exists() and not erase:
#                 print(f"File {destination} already exists! Use erase option to overwrite.")
#             else:
#                 with gzip.open(destination, "wb") as f:
#                     pickle.dump(self, f)
#         else:
#             destination = destination.with_suffix(".pkl")
#             if destination.exists() and not erase:
#                 print(f"File {destination} already exists! Use erase option to overwrite.")
#             else:
#                 with open(destination, "wb") as f:
#                     pickle.dump(self, f)
#
#     def load(self, filename: str, path='.'):
#         """
#         Load instance from file.
#
#         Parameters
#         ----------
#         filename: str
#             The stem of the filename.
#         path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
#             The location path.
#         """
#         path = Path(path)
#         dest = path / Path(filename).with_suffix(".pkl")
#         if dest.exists():
#             with open(dest, 'rb') as f:
#                 self.__dict__.update(pickle.load(f).__dict__)
#         else:
#             dest = dest.with_suffix('.pkl.gz')
#             if dest.exists():
#                 with gzip.open(dest) as f:
#                     self.__dict__.update(pickle.load(f).__dict__)
#             else:
#                 raise FileNotFoundError(
#                     errno.ENOENT, os.strerror(errno.ENOENT), dest)
#
#
# def default_preprocessor(txt):
#     """
#     Default string preprocessor: trim extra spaces and lower case from string `txt`.
#
#     Parameters
#     ----------
#     txt: :py:class:`str`
#         Text to process.
#
#     Returns
#     -------
#     :py:class:`str`
#         Processed text.
#
#     Examples
#     ---------
#     >>> default_preprocessor(" LaTeX RuleZ    ")
#     'latex rulez'
#     """
#     return txt.strip().lower()
#
#
# @njit
# def set_seed(seed=42):
#     """
#     Set the seed of the numba random generator (which is distinct from the numpy random generator).
#
#     Parameters
#     ----------
#     seed: :py:class:`int`
#         The seed number.
#
#     Returns
#     -------
#     None
#     """
#     np.random.seed(seed)
#
#
# def make_random_bool_generator(probability_true=.5):
#     """
#     Provides a (possibly biased) random generator of booleans.
#
#     Parameters
#     ----------
#     probability_true: :py:class:`float`, optional.
#         Probability to return `True`.
#
#     Returns
#     -------
#     random_boolean: callable
#         A function that returns a random boolean when called.
#
#     Examples
#     --------
#     >>> rb = make_random_bool_generator()
#     >>> set_seed(seed=42)
#     >>> [rb() for _ in range(10)]
#     [True, False, False, False, True, True, True, False, False, False]
#     """
#     @njit
#     def rb():
#         return np.random.rand() < probability_true
#
#     return rb
