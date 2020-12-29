import numpy as np
from scipy.sparse import coo_matrix

from .common import default_preprocessor, make_random_bool_generator, set_seed, MixInIO
from bof.cython.count import c_fit_transform


def number_of_factors(length, n_range=None):
    """
    Return the number of factors (with multiplicity) of size at most `n_range` that exist in a text of length `length`.
    This allows to pre-allocate working memory.

    Parameters
    ----------
    length: :py:class:`int`
        Length of the text.
    n_range: :py:class:`int` or None
        Maximal factor size. If `None`, all factors are considered.

    Returns
    -------
    int
        The number of factors (with multiplicity).

    Examples
    --------
    >>> l = len("riri")
    >>> number_of_factors(l)
    10
    >>> number_of_factors(l, n_range=2)
    7
    """
    if n_range is None or n_range > length:
        return length * (length + 1) // 2
    return n_range * (length - n_range) + n_range * (n_range + 1) // 2


def build_end(n_range=None):
    """
    Return a function of a starting position `s` and a text length `l` that tells the end of scanning text from `s`.
    It avoids to test the value of n_range all the time when doing factor extraction.

    Parameters
    ----------
    n_range: :py:class:`int` or None
         Maximal factor size. If `None`, all factors are considered.

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
        return lambda s, l: min(s + n_range, l)
    else:
        return lambda s, l: l


class CountVectorizer(MixInIO):
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
    features: :py:class:`list` of :py:class:`str`
        List of factors.
    features_: :py:class:`dict` of :py:class:`str` -> :py:class:`int`
        Dictionary that maps factors to their index in the list.
    m: :py:class:`int`
        Number of factors.

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
    def __init__(self, n_range=5, preprocessor=None, filename=None, path='.'):
        if filename is not None:
            self.load(filename=filename, path=path)
        else:
            self.features_ = dict()
            self.n_range = n_range
            if preprocessor is None:
                preprocessor = default_preprocessor
            self.preprocessor = preprocessor

    @property
    def m(self):
        return len(self.features_)

    @property
    def features(self):
        return list(self.features_)

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
        """
        if reset:
            self.features_ = dict()

        if self.n_range is None:
            use_range = False
            n_range = 0
        else:
            use_range = True
            n_range = self.n_range

        return c_fit_transform(corpus, self.features_, self.preprocessor, n_range, use_range)


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
            self.features_ = dict()
        end = build_end(self.n_range)
        for i, txt in enumerate(corpus):
            txt = self.preprocessor(txt)
            length = len(txt)
            for start in range(length):
                f = ""
                for letter in txt[start:end(start, length)]:
                    f += letter
                    self.features_.setdefault(f, len(self.features_))

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
        tot_size = sum(number_of_factors(len(txt.strip().lower()), self.n_range) for txt in corpus)
        feature_indices = np.zeros(tot_size, dtype=np.uintc)
        document_indices = np.zeros(tot_size, dtype=np.uintc)
        ptr = 0
        end = build_end(self.n_range)
        for i, txt in enumerate(corpus):
            start_ptr = ptr
            txt = self.preprocessor(txt)
            length = len(txt)
            for start in range(length):
                f = ""
                for letter in txt[start:end(start, length)]:
                    f += letter
                    if f in self.features_:
                        feature_indices[ptr] = self.features_[f]
                        ptr += 1
            document_indices[start_ptr:ptr] = i

        feature_indices = feature_indices[:ptr]
        document_indices = document_indices[:ptr]

        return coo_matrix((np.ones(ptr, dtype=np.uintc), (document_indices, feature_indices)),
                          shape=(len(corpus), self.m)).tocsr()

    def sampling_fit(self, corpus, reset=True, sampling_rate=.5):
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

        >>> set_seed(2012)
        >>> vectorizer.sampling_fit(["riri", "fifi", "rififi"])
        >>> vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi', 'if', 'ifi', 'ifif', 'ififi']

        We random fit another corpus. We reset the seed to reproduce the example above.

        >>> set_seed(2012)
        >>> vectorizer.sampling_fit(["riri", "fifi"])

        The factors have been implicitly reset.

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi']

        We add another corpus to the fit by setting `reset` to `False`:

        >>> vectorizer.sampling_fit(["rififi"], reset=False)

        The list of features has been updated:

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi', 'if', 'ifi', 'ifif', 'ififi']
        """
        if reset:
            self.features_ = dict()
        rb = make_random_bool_generator(probability_true=sampling_rate)
        end = build_end(self.n_range)
        for i, txt in enumerate(corpus):
            txt = self.preprocessor(txt)
            length = len(txt)
            for start in range(length):
                if rb():
                    f = ""
                    for letter in txt[start:end(start, length)]:
                        f += letter
                        self.features_.setdefault(f, len(self.features_))
