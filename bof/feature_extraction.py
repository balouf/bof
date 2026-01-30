from bof.numba import (
    jit_fit_transform,
    jit_fit,
    jit_transform,
    jit_sampling_fit,
    empty_features,
    default_preprocessor,
)
from numba.typed import List as TypedList
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np


def to_typed_list(corpus):
    """
    Convert a Python list of strings to a Numba typed list.

    This avoids the deprecated 'reflected list' type warning when passing
    Python lists to JIT-compiled functions.

    If the input is already a Numba typed list, it is returned as-is.

    Parameters
    ----------
    corpus: :py:class:`list` of :py:class:`str`
        List of strings to convert.

    Returns
    -------
    :class:`numba.typed.List`
        Typed list of strings.
    """
    if isinstance(corpus, TypedList):
        return corpus
    typed_corpus = TypedList()
    for txt in corpus:
        typed_corpus.append(txt)
    return typed_corpus


def build_end(n_range=None):
    """
    Return a function of a starting position `s` and a text length `tl` that tells the end of scanning text from `s`.
    It avoids to test the value of n_range all the time when doing factor extraction.

    Parameters
    ----------
    n_range: :py:class:`int` or None
         Maximal factor size. If 0 or `None`, all factors are considered.

    Returns
    -------
    callable

    Examples
    --------
    >>> end = build_end()
    >>> end(7, 15)
    15
    >>> end(13, 15)
    15
    >>> end = build_end(5)
    >>> end(7, 15)
    12
    >>> end(13, 15)
    15

    """
    if n_range:
        return lambda s, tl: min(s + n_range, tl)
    else:
        return lambda s, tl: tl


class CountVectorizer:
    """
    Counts the factors of a list of document.

    Parameters
    ----------
    preprocessor: callable, optional
        Preprocessing function to apply to texts before adding them to the factor tree.
    n_range: :py:class:`int` or None, optional
        Maximum factor size. If `None`, all factors will be extracted.
    filename: :py:class:`str`, optional
        If set, load from corresponding file.
    path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
        If set, specify the directory where the file is located.

    Attributes
    ----------
    features_: :py:class:`dict` of :py:class:`str` -> :py:class:`int`
        Dictionary that maps factors to their index in the list.

    Examples
    --------

    Build a vectorizer limiting factor size to 3:

    >>> vectorizer = CountVectorizer(n_range=3)

    Build the factor matrix of a corpus of texts.

    >>> corpus = ["riri", "fifi", "rififi"]
    >>> vectorizer.fit_transform(corpus=corpus).toarray() # doctest: +NORMALIZE_WHITESPACE
    array([[2, 2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 2, 0, 0, 2, 2, 1, 1, 1, 0],
           [1, 1, 0, 3, 0, 0, 2, 2, 1, 2, 2, 1]], dtype=uint32)

    List the factors in the corpus:

    >>> vectorizer.features
    ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
    """

    def __init__(self, n_range=5, preprocessor=None):
        self.features_ = empty_features()  # dict()
        self.n_range = n_range
        if preprocessor is None:
            preprocessor = default_preprocessor
        self.preprocessor = preprocessor

    @property
    def m(self):
        """
        Get the number of features.

        Returns
        -------
        m: :py:class:`int`
            Number of factors.
        """
        return len(self.features_)

    @property
    def features(self):
        """
        Get the list of features (internally, features are stored as a Numba Typed :py:class:`dict`
        that associates factors to indexes).

        Returns
        -------
        features: :py:class:`list` of :py:class:`str`
            List of factors.

        """
        return list(self.features_)

    def no_none_range(self):
        """
        Replace None n_range by 0 before passing to cython code

        Returns
        -------
        None
        """
        if self.n_range is None:
            self.n_range = 0

    def fit_transform(self, corpus, reset=True):
        """
        Build the features and return the factor matrix.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`str`.
            Texts to analyze.
        reset: :py:class:`bool`, optional
            Clears factors. If False, factors are updated instead.

        Returns
        -------
        :class:`~scipy.sparse.csr_matrix`
            A sparse matrix that indicates for each document of the corpus its factors and their multiplicity.

        Examples
        --------

        Build a FactorTree from a corpus of three documents:

        >>> vectorizer = CountVectorizer(n_range=3)
        >>> vectorizer.fit_transform(["riri", "fifi", "rififi"]).toarray() # doctest: +NORMALIZE_WHITESPACE
        array([[2, 2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 2, 0, 0, 2, 2, 1, 1, 1, 0],
               [1, 1, 0, 3, 0, 0, 2, 2, 1, 2, 2, 1]], dtype=uint32)

        List of factors (of size at most 3):

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

        Build a FactorTree from a corpus of two documents.

        >>> vectorizer.fit_transform(["fifi", "rififi"]).toarray() # doctest: +NORMALIZE_WHITESPACE
        array([[2, 2, 1, 2, 1, 1, 0, 0, 0],
               [2, 2, 1, 3, 2, 2, 1, 1, 1]], dtype=uint32)

        Notice the implicit reset, as only factors from "fifi" and "rififi" are present:

        >>> vectorizer.features
        ['f', 'fi', 'fif', 'i', 'if', 'ifi', 'r', 'ri', 'rif']

        >>> vectorizer.m
        9

        With `reset` set to `False`, we can add another list without discarding pre-existing factors.

        >>> vectorizer.fit_transform(["riri"], reset=False).toarray() # doctest: +NORMALIZE_WHITESPACE
        array([[0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 1, 1]], dtype=uint32)

        Notice the presence of empty columns, which corresponds to pre-existing factors that do not exist in "riri".

        The size and list of factors:

        >>> vectorizer.m
        12

        >>> vectorizer.features
        ['f', 'fi', 'fif', 'i', 'if', 'ifi', 'r', 'ri', 'rif', 'rir', 'ir', 'iri']

        Setting n_range to None will compute all factors.

        >>> vectorizer.n_range = None
        >>> vectorizer.fit_transform(["riri", "fifi", "rififi"]).toarray() # doctest: +NORMALIZE_WHITESPACE
        array([[2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 2, 0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
               [1, 1, 0, 0, 3, 0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1]], dtype=uint32)


        """
        if reset:
            self.features_ = empty_features()  # dict()

        self.no_none_range()

        typed_corpus = to_typed_list(corpus)
        tot_size, document_indices, feature_indices, m = jit_fit_transform(
            corpus=typed_corpus,
            preprocessor=self.preprocessor,
            features=self.features_,
            n_range=self.n_range,
        )
        return csr_matrix(
            coo_matrix(
                (
                    np.ones(tot_size, dtype=np.uintc),
                    (document_indices, feature_indices),
                ),
                shape=(len(corpus), m),
            ),
            dtype=np.uint32,
        )

    def fit(self, corpus, reset=True):
        """
        Build the features. Does not build the factor matrix.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`str`.
            Texts to analyze.
        reset: :py:class:`bool`
            Clears current features and corpus. Features will be updated instead.

        Returns
        -------
        None

        Examples
        --------

        We compute the factors of a corpus.

        >>> vectorizer = CountVectorizer(n_range=3)
        >>> vectorizer.fit(["riri", "fifi", "rififi"])

        The `fit` method does not return anything, but the factors have been populated:

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

        We fit another corpus.

        >>> vectorizer.fit(["riri", "fifi"])

        The factors have been implicitly reset (`rif` is gone in this toy example):

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi']

        We keep pre-existing factors by setting `reset` to `False`:

        >>> vectorizer.fit(["rififi"], reset=False)

        The list of features has been updated (with `rif``):

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        """
        if reset:
            self.features_ = empty_features()  # dict()

        self.no_none_range()

        typed_corpus = to_typed_list(corpus)
        jit_fit(
            corpus=typed_corpus,
            preprocessor=self.preprocessor,
            features=self.features_,
            n_range=self.n_range,
        )

    def transform(self, corpus):
        """
        Build factor matrix from the factors already computed. New factors are discarded.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`str`.
            Texts to analyze.

        Returns
        -------
        :class:`~scipy.sparse.csr_matrix`
            The factor count of the input corpus NB: if reset is set to `False`, the factor count of the pre-existing
            corpus is not returned but is internally preserved.

        Examples
        --------

        To start, we fit a corpus:

        >>> vectorizer = CountVectorizer(n_range=3)
        >>> vectorizer.fit_transform(["riri", "fifi", "rififi"]).toarray() # doctest: +NORMALIZE_WHITESPACE
        array([[2, 2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 2, 0, 0, 2, 2, 1, 1, 1, 0],
               [1, 1, 0, 3, 0, 0, 2, 2, 1, 2, 2, 1]], dtype=uint32)

        The factors are:

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

        We now apply a transform.

        >>> vectorizer.transform(["fir", "rfi"]).toarray() # doctest: +NORMALIZE_WHITESPACE
        array([[1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
               [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=uint32)


        The features have not been updated. For example, the only factors reported for "rfi" are "r", "i", "f", and
        "fi". Factors that were not fit (e.g. `rf`) are discarded.
        """
        self.no_none_range()

        typed_corpus = to_typed_list(corpus)
        tot_size, document_indices, feature_indices, m = jit_transform(
            corpus=typed_corpus,
            preprocessor=self.preprocessor,
            features=self.features_,
            n_range=self.n_range,
        )
        return csr_matrix(
            coo_matrix(
                (
                    np.ones(tot_size, dtype=np.uintc),
                    (document_indices, feature_indices),
                ),
                shape=(len(corpus), m),
            ),
            dtype=np.uint32,
        )

    def sampling_fit(self, corpus, reset=True, sampling_rate=0.5, seed=None):
        """
        Build a partial factor tree where only a random subset of factors are selected. Note that there is no
        `sampling_fit_transform` method, as mutualizing the processes would introduce incoherences in the factor
        description: you have to do a `sampling_fit` followed by a `transform`.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`str`
            Texts to analyze.
        reset: :py:class:`bool`
            Clears FactorTree. If False, FactorTree will be updated instead.
        sampling_rate: :py:class:`float`
            Probability to explore factors starting from one given position in the text.
        seed: :py:class:`int`, optional
            Seed of the random generator.

        Returns
        -------
        None

        Examples
        --------

        We fit a corpus to a tree a normal way to see the complete list of factors of size at most 3..

        >>> vectorizer = CountVectorizer()
        >>> vectorizer.fit(["riri", "fifi", "rififi"])
        >>> vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'fifi', 'if', 'ifi', 'rif', 'rifi', 'rifif', 'ifif', 'ififi']

        Now we use a sampling fit instead. Only a subset of the factors are selected.

        >>> vectorizer.sampling_fit(["riri", "fifi", "rififi"], seed=42)
        >>> vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'f', 'fi', 'fif', 'fifi', 'i', 'if', 'ifi']

        We random fit another corpus. We reset the seed to reproduce the example above.

        >>> vectorizer.sampling_fit(["riri", "fifi"], sampling_rate=.2)

        The factors have been implicitly reset.

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri']

        We add another corpus to the fit by setting `reset` to `False`:

        >>> vectorizer.sampling_fit(["rififi"], reset=False, sampling_rate=.2)

        The list of features has been updated:

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi']
        """
        if reset:
            self.features_ = empty_features()  # dict()

        self.no_none_range()

        typed_corpus = to_typed_list(corpus)
        jit_sampling_fit(
            corpus=typed_corpus,
            preprocessor=self.preprocessor,
            features=self.features_,
            n_range=self.n_range,
            rate=sampling_rate,
            seed=seed,
        )
