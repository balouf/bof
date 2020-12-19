import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack

from .common import default_preprocessor


def number_of_factors(length, n_range=None):
    """
    Return the number of factors (with multiplicity) of size at most `n_range` that exist in a text of length `length`.

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


class CountVectorizer:
    """
    Counts the factors of a list of document.

    Parameters
    ----------
    corpus: :py:class:`list` of :py:class:`str`, optional
        Corpus of documents to decompose into factors.
    preprocessor: callable, optional
        Preprocessing function to apply to texts before adding them to the factor tree.
    n_range: :py:class:`int` or None, optional
        Maximum factor size. If `None`, all factors will be extracted.

    Attributes
    ----------
    feats_to_docs: :class:`~scipy.sparse.csr_matrix`
        matrix features X documents
    docs_to_feats: :class:`~scipy.sparse.csr_matrix`
        matrix documents X features
    corpus: :py:class:`list` of :py:class:`srt`
        The list of documents.
    features: :py:class:`list` of :py:class:`str`
        List of factors.
    features_: :py:class:`dict` of :py:class:`str` -> :py:class:`int`
        Dictionary that maps factors to their index in the list.
    m: :py:class:`int`
        Number of factors.

    Examples
    --------

    Build a vectorizer from a corpus of texts,limiting factor size to 3:

    >>> corpus = ["riri", "fifi", "rififi"]
    >>> vectorizer = CountVectorizer(corpus=corpus, n_range=3)

    List the number of unique factors for each text:

    >>> vectorizer.self_factors()
    array([6, 6, 9], dtype=int32)

    List the factors in the corpus:

    >>> vectorizer.features
    ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

    Display the factors per document:

    >>> print(vectorizer.tostr())
    riri: 'r'x2, 'ri'x2, 'rir'x1, 'i'x2, 'ir'x1, 'iri'x1
    fifi: 'i'x2, 'f'x2, 'fi'x2, 'fif'x1, 'if'x1, 'ifi'x1
    rififi: 'r'x1, 'ri'x1, 'i'x3, 'f'x2, 'fi'x2, 'fif'x1, 'if'x2, 'ifi'x2, 'rif'x1
    """

    def __init__(self, corpus=None, n_range=5, preprocessor=None):
        self.feats_to_docs = csr_matrix((0, 0), dtype=np.uint64)
        self.docs_to_feats = csr_matrix((0, 0), dtype=np.uint64)
        self.m = 0
        self.features_ = dict()
        self.features = list()
        self.corpus = list()
        self.n_range = n_range
        if preprocessor is None:
            preprocessor = default_preprocessor
        self.preprocessor = preprocessor
        if corpus is not None:
            self.fit_transform(corpus)

    def fit_transform(self, corpus, reset=True):
        """
        Build the features and the factor matrices.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`str`.
            Texts to analyze.
        reset: :py:class:`bool`
            Clears FactorTree. If False, FactorTree will be updated instead.

        Returns
        -------
        docs_to_feats: :class:`~scipy.sparse.csr_matrix`

        Examples
        --------

        Build a FactorTree from a corpus of three documents:

        >>> vectorizer = CountVectorizer(n_range=3)
        >>> vectorizer.fit_transform(["riri", "fifi", "rififi"]) # doctest: +NORMALIZE_WHITESPACE
        <3x12 sparse matrix of type '<class 'numpy.uint64'>'
            with 21 stored elements in Compressed Sparse Row format>

        List of documents:

        >>> vectorizer.corpus
        ['riri', 'fifi', 'rififi']

        List of factors (of size at most 3):

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

        Build a FactorTree from a corpus of two documents.

        >>> vectorizer.fit_transform(["riri", "fifi"]) # doctest: +NORMALIZE_WHITESPACE
        <2x11 sparse matrix of type '<class 'numpy.uint64'>'
            with 12 stored elements in Compressed Sparse Row format>

        Notice the implicit reset, as there are now two documents:

        >>> vectorizer.corpus
        ['riri', 'fifi']

        Similarly, the factors are these from ``riri`` and ``fifi``.

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi']

        >>> vectorizer.m
        11

        With `reset` set to `False`, we can add another list while keeping the previous state.

        >>> vectorizer.fit_transform(["rififi"], reset=False) # doctest: +NORMALIZE_WHITESPACE
        <3x12 sparse matrix of type '<class 'numpy.uint64'>'
            with 21 stored elements in Compressed Sparse Row format>

        We have now 2+1=3 documents.

        >>> vectorizer.corpus
        ['riri', 'fifi', 'rififi']

        The list of features has been updated as well:

        >>> vectorizer.m
        12

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        """
        if reset:
            self.feats_to_docs = csr_matrix((0, 0), dtype=np.uint64)
            self.docs_to_feats = csr_matrix((0, 0), dtype=np.uint64)
            self.m = 0
            self.features_ = dict()
            self.features = list()
            self.corpus = list()
        old_n = len(self.corpus)
        new_n = len(corpus)
        self.corpus += corpus
        tot_size = sum(number_of_factors(len(txt.strip().lower()), self.n_range) for txt in corpus)
        feature_indices = np.zeros(tot_size, dtype=np.uint64)
        document_indices = np.zeros(tot_size, dtype=np.uint64)
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
                    else:
                        self.features_[f] = self.m
                        self.features.append(f)
                        feature_indices[ptr] = self.m
                        self.m += 1
                    ptr += 1
            document_indices[start_ptr:ptr] = i

        new_count = coo_matrix((np.ones(tot_size, dtype=np.uint64), (feature_indices, document_indices)),
                               shape=(self.m, new_n))
        self.feats_to_docs.resize(self.m, old_n)
        self.feats_to_docs = hstack([self.feats_to_docs, new_count.tocsr()])
        self.docs_to_feats.resize(old_n, self.m)
        self.docs_to_feats = vstack([self.docs_to_feats, new_count.T.tocsr()])
        return self.docs_to_feats

    def fit(self, corpus, reset=True):
        """
        Build the features. Does not update factor matrices.

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

        The inner corpus remains empty:

        >>> vectorizer.corpus
        []

        The factors have been populated:

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

        We fit another corpus.

        >>> vectorizer.fit(["riri", "fifi"])

        The inner corpus remains empty:

        >>> vectorizer.corpus
        []

        The factors have been implicitly reset and updated from the new corpus (`rif` is gone in this toy example):

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi']

        We add another corpus to the fit by setting `reset` to `False`:

        >>> vectorizer.fit(["rififi"], reset=False)

        The inner corpus remains empty:

        >>> vectorizer.corpus
        []

        The list of features has been updated (with `rif``):

        >>> vectorizer.features
        ['r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        """
        if reset:
            self.feats_to_docs = csr_matrix((0, 0), dtype=np.uint64)
            self.docs_to_feats = csr_matrix((0, 0), dtype=np.uint64)
            self.m = 0
            self.features_ = dict()
            self.features = list()
            self.corpus = list()
        end = build_end(self.n_range)
        for i, txt in enumerate(corpus):
            txt = self.preprocessor(txt)
            length = len(txt)
            for start in range(length):
                f = ""
                for letter in txt[start:end(start, length)]:
                    f += letter
                    if f not in self.features_:
                        self.features_[f] = self.m
                        self.features.append(f)
                        self.m += 1

    def transform(self, corpus, reset=True):
        """
        Build factor matrices from the factors already computed. New factors are discarded.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`str`.
            Texts to analyze.
        reset: :py:class:`bool`, optional
            Clears internal corpus. If False, internal corpus will be updated instead.

        Returns
        -------
        :class:`~scipy.sparse.csr_matrix`
            The factor count of the input corpus NB: if reset is set to `False`, the factor count of the pre-existing
            corpus is not returned but is internally preserved.

        Examples
        --------

        To start, we fit_transform a corpus:

        >>> vectorizer = CountVectorizer(n_range=3)
        >>> vectorizer.fit_transform(["riri", "fifi", "rififi"]) # doctest: +NORMALIZE_WHITESPACE
        <3x12 sparse matrix of type '<class 'numpy.uint64'>'
            with 21 stored elements in Compressed Sparse Row format>

        The corpus decomposition in factors:

        >>> print(vectorizer.tostr())
        riri: 'r'x2, 'ri'x2, 'rir'x1, 'i'x2, 'ir'x1, 'iri'x1
        fifi: 'i'x2, 'f'x2, 'fi'x2, 'fif'x1, 'if'x1, 'ifi'x1
        rififi: 'r'x1, 'ri'x1, 'i'x3, 'f'x2, 'fi'x2, 'fif'x1, 'if'x2, 'ifi'x2, 'rif'x1

        We now apply a transform.

        >>> vectorizer.transform(["fir", "rfi"]) # doctest: +NORMALIZE_WHITESPACE
        <2x12 sparse matrix of type '<class 'numpy.uint64'>'
            with 9 stored elements in Compressed Sparse Row format>

        Observe the corpus decomposition: the old corpus has been erased, and new factors (e.g. `rf`) are discarded.

        >>> print(vectorizer.tostr())
        fir: 'r'x1, 'i'x1, 'ir'x1, 'f'x1, 'fi'x1
        rfi: 'r'x1, 'i'x1, 'f'x1, 'fi'x1

        We update (without discarding previous entries) the corpus with a new one. Note that only the matrix of the
        update is returned.

        >>> vectorizer.transform(["rififi"], reset=False) # doctest: +NORMALIZE_WHITESPACE
        <1x12 sparse matrix of type '<class 'numpy.uint64'>'
            with 9 stored elements in Compressed Sparse Row format>

        >>> print(vectorizer.tostr())
        fir: 'r'x1, 'i'x1, 'ir'x1, 'f'x1, 'fi'x1
        rfi: 'r'x1, 'i'x1, 'f'x1, 'fi'x1
        rififi: 'r'x1, 'ri'x1, 'i'x3, 'f'x2, 'fi'x2, 'fif'x1, 'if'x2, 'ifi'x2, 'rif'x1
        """
        if reset:
            self.feats_to_docs = csr_matrix((0, 0), dtype=np.uint64)
            self.docs_to_feats = csr_matrix((0, 0), dtype=np.uint64)
            self.corpus = list()
        old_n = len(self.corpus)
        new_n = len(corpus)
        self.corpus += corpus
        tot_size = sum(number_of_factors(len(txt.strip().lower()), self.n_range) for txt in corpus)
        feature_indices = np.zeros(tot_size, dtype=np.uint64)
        document_indices = np.zeros(tot_size, dtype=np.uint64)
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

        new_count = coo_matrix((np.ones(ptr, dtype=np.uint64), (feature_indices, document_indices)),
                               shape=(self.m, new_n))
        self.feats_to_docs.resize(self.m, old_n)
        self.feats_to_docs = hstack([self.feats_to_docs, new_count.tocsr()])

        new_d2f = new_count.T.tocsr()
        self.docs_to_feats.resize(old_n, self.m)
        self.docs_to_feats = vstack([self.docs_to_feats, new_d2f])
        return new_d2f

    def self_factors(self):
        return self.docs_to_feats.indptr[1:] - self.docs_to_feats.indptr[0:-1]

    def tostr(self):
        """
        Simple side function to check the factor decomposition on a small corpus. DO NOT USE ON LARGE CORPI!

        Returns
        -------
        str

        Examples
        --------

        Analyzes the factor of size at most 3 of `["riri", "fifi", "rififi"]`:

        >>> vectorizer = CountVectorizer(["riri", "fifi", "rififi"], n_range=3)
        >>> print(vectorizer.tostr())
        riri: 'r'x2, 'ri'x2, 'rir'x1, 'i'x2, 'ir'x1, 'iri'x1
        fifi: 'i'x2, 'f'x2, 'fi'x2, 'fif'x1, 'if'x1, 'ifi'x1
        rififi: 'r'x1, 'ri'x1, 'i'x3, 'f'x2, 'fi'x2, 'fif'x1, 'if'x2, 'ifi'x2, 'rif'x1
        """
        d2f = self.docs_to_feats

        def doc_factors(i):
            factors = ", ".join([f"'{self.features[j]}'x{d2f[i, j]}" for j in
                                 d2f.indices[d2f.indptr[i]:d2f.indptr[i + 1]]])
            return f"{self.corpus[i]}: {factors}"

        return "\n".join(doc_factors(i) for i in range(len(self.corpus)))
