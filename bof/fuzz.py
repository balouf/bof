import numpy as np
from numba import njit, prange

from bof.feature_extraction import CountVectorizer


@njit(cache=True, parallel=True)
def jit_common_factors(queries_length, xind, xptr, choices_length, yind, yptr, m):
    """
    Jitted function to compute the common factors between a corpus of queries and a corpus of choices.

    Parameters
    ----------
    queries_length: :py:class:`int`
        Number of documents in the corpus of queries
    xind: :class:`~numpy.ndarray`
        Indices of the query factor matrix
    xptr: :class:`~numpy.ndarray`
        pointers of the query factor matrix
    choices_length: :py:class:`int`
        Number of documents in the corpus of choices
    yind: :class:`~numpy.ndarray`
        Indices of the transposed choices factor matrix
    yptr: :class:`~numpy.ndarray`
        Pointers of the transposed choices factor matrix
    m: :py:class:`int`
        Size of the factor space for choices

    Returns
    -------
    :class:`~numpy.ndarray`
        A `queries_length X choices_length` matrix that contains the number of (unique) factors between choices and
        queries.
    """
    res = np.zeros((queries_length, choices_length), dtype=np.int32)
    for i in prange(queries_length):
        for k in xind[xptr[i] : xptr[i + 1]]:
            if (
                k < m
            ):  # NB with sorted_indices, could be a break, but no guarantee enforced
                for j in yind[yptr[k] : yptr[k + 1]]:
                    res[i, j] += 1
    return res


@njit(cache=True, parallel=True)
def jit_jc(
    queries_factors, choices_factors, common_factors, length_impact, threshold=0.0
):
    """
    Jitted function to compute a joint complexity between a corpus of queries and a corpus of choices.

    Parameters
    ----------
    queries_factors: :class:`~numpy.ndarray`
        Vector of the number of unique factors for each query.
    choices_factors: :class:`~numpy.ndarray`
        Vector of the number of unique factors for each choice.
    common_factors: :class:`~numpy.ndarray`
        Matrix of the number of common unique factors between queries and choices.
    length_impact: :py:class:`float`
        Importance of the length difference between two texts when computing the scores.
    threshold: :class:`float`
        Don't compute JC is common factors is less than threshold X (# query factors)

    Returns
    -------
    :class:`~numpy.ndarray`
        Joint Complexity matrix
    """
    res = np.zeros((len(queries_factors), len(choices_factors)))
    for i in prange(len(queries_factors)):
        query_factors = queries_factors[i]
        for j in range(len(choices_factors)):
            cf = common_factors[i, j]
            choice_factors = choices_factors[j]
            if cf > query_factors * threshold:
                if query_factors < choice_factors:
                    renorm = 2 * (
                        length_impact * choice_factors
                        + (1 - length_impact) * query_factors
                    )
                else:
                    renorm = 2 * (
                        length_impact * query_factors
                        + (1 - length_impact) * choice_factors
                    )
                res[i, j] = 100 * cf / (renorm - cf)
    return res


@njit(cache=True, parallel=True)
def jit_square_factors(xind, xptr, yind, yptr, n, length_impact):
    """
    Jitted function to compute the joint complexity between texts of a corpus.

    Parameters
    ----------
    xind: :class:`~numpy.ndarray`
        Indices of the factor matrix
    xptr: :class:`~numpy.ndarray`
        pointers of the factor matrix
    yind: :class:`~numpy.ndarray`
        Indices of the transposed factor matrix
    yptr: :class:`~numpy.ndarray`
        Pointers of the transposed factor matrix
    n: :py:class:`int`
        Corpus size
    length_impact: :py:class:`float`
        Importance of the length difference between two texts when computing the scores.

    Returns
    -------
    :class:`~numpy.ndarray`
        A `n X n` matrix that contains joint complexity scores of the corpus.
    """
    self_factors = xptr[1:] - xptr[:-1]
    common_factors = np.zeros((n, n), dtype=np.int32)
    for i in prange(n):
        for k in xind[xptr[i] : xptr[i + 1]]:
            for j in yind[yptr[k] : yptr[k + 1]]:
                common_factors[i, j] += 1
    return jit_jc(self_factors, self_factors, common_factors, length_impact)


def txt_max(txt_list):
    """
    Auxiliary function that returns the index of the longest text from a list. In case of ties, alphabetic order is used.

    Parameters
    ----------
    txt_list: :py:class:`list` of :py:class:`str`
        List of texts.

    Returns
    -------
    :py:class:`int`
        The index of the longest text.

    Examples
    --------

    >>> txt_max(["This is long!", "That is long!", "It's short"])
    1
    """
    return min(zip(range(len(txt_list)), txt_list), key=lambda z: (-len(z[1]), z[1]))[0]


def get_best_choices(choices, scores, limit):
    """
    Given a list of choices with scores, extract the best choices.

    Parameters
    ----------
    choices: :py:class:`list` of :py:class:`str`
        List of choices.
    scores: :class:`~numpy.ndarray`
        Score of the choices (must have the same size as `choices`)
    limit: :py:class:`int` or None
        Maximal number of results to give. If None, all choices will be returned (sorted).

    Returns
    -------
    :py:class:`list`
        List of tuples containing the choices and their scores.
    """
    if limit is None:
        sorted_top = np.argsort(-scores)
        # return sorted(zip(choices, scores), key=lambda i: i[1], reverse=True)
    else:
        unsorted_top = np.argpartition(scores, -limit)[-limit:]
        sorted_top = unsorted_top[np.argsort(-scores[unsorted_top])]
    return [(choices[i], scores[i]) for i in sorted_top]
    # return heapq.nlargest(limit, zip(choices, scores), key=lambda i: i[1])


def get_best_choice(choices, scores, score_cutoff):
    """
    Given a list of choices with scores, extract the best choice.

    Parameters
    ----------
    choices: :py:class:`list` of :py:class:`str`
        List of choices.
    scores: :class:`~numpy.ndarray`
        Score of the choices (must have the same size as `choices`)
    score_cutoff: :py:class:`float`
        Minimal score that the result must achieve to be considered a match

    Returns
    -------
    :py:class:`tuple` or None
        Tuple containing the choice and its score if the latter is above the cutoff, None otherwise.
    """
    index = np.argmax(scores)
    if scores[index] > score_cutoff:
        return choices[index], scores[index]


class Process:
    """
    The process class is used to compute the closest choices from a list of queries base on joint complexity.

    Parameters
    ----------
    n_range: :py:class:`int` or None, optional
        Maximum factor size. If `None`, all factors will be extracted.:
    preprocessor: callable, optional
        Preprocessing function to apply to texts before adding them to the factor tree.
    length_impact: :py:class:`float`
        Importance of the length difference between two texts when computing the scores.
    allow_updates: :py:class:`bool`
        When transforming queries, are new factors kept in the :py:class:`~bof.feature_extraction.CountVectorizer`
    filename: :py:class:`str`, optional
        If set, load from corresponding file.
    path: :py:class:`str` or :py:class:`~pathlib.Path`, optional
        If set, specify the directory where the file is located.

    Attributes
    ----------
    vectorizer: :py:class:`~bof.feature_extraction.CountVectorizer`
        The vectorizer used to compute factors.
    choices: :py:class:`list` of :py:class:`str`
        The possible choices.
    choices_matrix: :class:`~scipy.sparse.csr_matrix`
        The factor matrix of the choices.
    choices_factors: :class:`~numpy.ndarray`
        Number of unique factors of each choice
    choices_length: :py:class:`int`
        Number of choices
    """

    def __init__(
        self, n_range=5, preprocessor=None, length_impact=0.5, allow_updates=True
    ):
        self.length_impact = length_impact
        self.allow_updates = allow_updates
        self.vectorizer = CountVectorizer(n_range=n_range, preprocessor=preprocessor)

        self.choices = None
        self.choices_matrix = None
        self.choices_factors = None
        self.choices_length = None

    def reset(self):
        """
        Clear choices from the object.

        Returns
        -------

        Examples
        --------

        >>> p = Process()
        >>> p.fit(["riri", "fifi", "rififi"])
        >>> p.choices
        ['riri', 'fifi', 'rififi']
        >>> p.choices_factors.astype(int)
        array([ 7,  7, 14])
        >>> p.reset()
        >>> p.choices is None
        True
        >>> p.choices_factors is None
        True
        """
        self.choices = None
        self.choices_matrix = None
        self.choices_factors = None
        self.choices_length = None

    def fit(self, choices):
        """
        Compute the factors of a list of choices.

        Parameters
        ----------
        choices: :py:class:`list` of :py:class:`str`
        The possible choices.

        Returns
        -------
        None

        Examples
        --------
        >>> p = Process()
        >>> p.fit(["riri", "fifi", "rififi"])

        The choices:

        >>> p.choices
        ['riri', 'fifi', 'rififi']

        Number of unique factors for each choice:

        >>> p.choices_factors.astype(int)
        array([ 7,  7, 14])

        The matrix that associates factors to choices:

        >>> p.choices_matrix.toarray() # doctest: +NORMALIZE_WHITESPACE
        array([[2, 0, 1],
               [2, 0, 1],
               [1, 0, 0],
               [1, 0, 0],
               [2, 2, 3],
               [1, 0, 0],
               [1, 0, 0],
               [0, 2, 2],
               [0, 2, 2],
               [0, 1, 1],
               [0, 1, 1],
               [0, 1, 2],
               [0, 1, 2],
               [0, 0, 1],
               [0, 0, 1],
               [0, 0, 1],
               [0, 0, 1],
               [0, 0, 1]], dtype=uint32)

        The corresponding factors:

        >>> p.vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'fifi', 'if', 'ifi', 'rif', 'rifi', 'rifif', 'ifif', 'ififi']
        """
        self.choices = choices
        self.choices_matrix = self.vectorizer.fit_transform(choices)
        self.choices_factors = (
            self.choices_matrix.indptr[1:] - self.choices_matrix.indptr[0:-1]
        )
        self.choices_matrix = self.choices_matrix.T.tocsr()
        self.choices_length = len(choices)

    def transform(self, queries, threshold=0.0):
        """
        Compute the joint complexities of queries against choices,

        Parameters
        ----------
        queries: :py:class:`list` of :py:class:`str`
            The list of queries.
        threshold: :class:`float`
            Relative threshold below which joint complexity is not actually computed.

        Returns
        -------

        Examples
        ---------

        >>> p = Process()
        >>> p.fit(["riri", "fifi", "rififi"])

        Notice the number of factors:

        >>> p.vectorizer.m
        18

        >>> p.transform(["rir", "fido", "rafifi", "screugneuhneu"]) # doctest: +NORMALIZE_WHITESPACE
        array([[71.42857143,  9.09090909, 18.75      ],
               [ 6.25      , 21.42857143, 14.28571429],
               [ 9.09090909, 41.17647059, 34.7826087 ],
               [ 1.92307692,  0.        ,  1.69491525]])

        The factors have been augmented with the ones from the queries:

        >>> p.vectorizer.m
        79

        This is could be a memory issue if you keep entering very different queries. To keep the factors clean after
        a transform, set `allow_updates` to False.

        >>> p.allow_updates = False
        >>> p.fit(["riri", "fifi", "rififi"])
        >>> p.transform(["rir", "fido", "rafifi", "screugneuhneu"]) # doctest: +NORMALIZE_WHITESPACE
        array([[71.42857143,  9.09090909, 18.75      ],
               [ 6.25      , 21.42857143, 14.28571429],
               [ 9.09090909, 41.17647059, 34.7826087 ],
               [ 1.92307692,  0.        ,  1.69491525]])
        >>> p.vectorizer.m
        18
        >>> p.vectorizer.features
        ['r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'fifi', 'if', 'ifi', 'rif', 'rifi', 'rifif', 'ifif', 'ififi']
        """
        m, _ = self.choices_matrix.shape
        queries_matrix = self.vectorizer.fit_transform(queries, reset=False)
        queries_factors = queries_matrix.indptr[1:] - queries_matrix.indptr[0:-1]
        queries_length = len(queries)
        common_factor_matrix = jit_common_factors(
            queries_length,
            queries_matrix.indices,
            queries_matrix.indptr,
            self.choices_length,
            self.choices_matrix.indices,
            self.choices_matrix.indptr,
            m,
        )
        if not self.allow_updates:
            extra_entries = self.vectorizer.m - m
            for _ in range(extra_entries):
                self.vectorizer.features_.popitem()

        return jit_jc(
            queries_factors,
            self.choices_factors,
            common_factor_matrix,
            self.length_impact,
            threshold=threshold,
        )

    def extract(self, queries, choices=None, limit=5, score_cutoff=40.0):
        """
        Find the best matches amongst a list of choices.

        Parameters
        ----------
        queries: :py:class:`str` or :py:class:`list` of :py:class:`str`
            Text (or list of texts) to match amongst the queries.
        choices: :py:class:`list` of :py:class:`str`, optional
            Possible choices. If None, the previously used (or fitted) choices will be used.
        limit: :py:class:`int` or None
            Maximal number of results to give. If None, all choices will be returned (sorted).
        score_cutoff: :py:class:`float`, optional
            Minimal score that a result must achieve to be considered a match

        Returns
        -------
        :py:class:`list`
            If `queries` is a single text, the list of tuples containing the best choices and their scores.
            If `queries` is a list of texts, the list of list of tuples containing the best choices and their scores.

        Examples
        ---------
        >>> p = Process()
        >>> choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
        >>> p.extract("new york jets", choices, limit=2)
        [('New York Jets', 100.0), ('New York Giants', 46.835443037974684)]
        >>> p.extract("new york jets", choices, limit=None) # doctest: +NORMALIZE_WHITESPACE
        [('New York Jets', 100.0),
        ('New York Giants', 46.835443037974684),
        ('Atlanta Falcons', 0.0),
        ('Dallas Cowboys', 0.0)]
        >>> p.extract(["new york", "atlanta"], choices, limit=2, score_cutoff=0.0) # doctest: +NORMALIZE_WHITESPACE
        [[('New York Jets', 56.60377358490566), ('New York Giants', 47.61904761904762)],
        [('Atlanta Falcons', 37.28813559322034), ('New York Giants', 7.594936708860759)]]
        """
        if isinstance(queries, str):
            is_str = True
            queries = [queries]
        else:
            is_str = False
        if choices is not None:
            self.fit(choices)
        jc_array = self.transform(queries, threshold=score_cutoff / 100)
        if is_str:
            return get_best_choices(self.choices, jc_array[0, :], limit)
        else:
            return [
                get_best_choices(self.choices, jc_array[i, :], limit)
                for i in range(len(queries))
            ]

    def extractOne(self, queries, choices=None, score_cutoff=40.0):
        """
        Find the best match amongst a list of choices.

        Parameters
        ----------
        queries: :py:class:`str` or :py:class:`list` of :py:class:`str`
            Text (or list of texts) to match amongst the queries.
        choices: :py:class:`list` of :py:class:`str`, optional
            Possible choices. If None, the previously used (or fitted) choices will be used.
        score_cutoff: :py:class:`float`, optional
            Minimal score that a result must achieve to be considered a match

        Returns
        -------
        :py:class:`tuple` or :py:class:`list`
            If `queries` is a single text, a tuple containing the best choice and its score.
            If `queries` is a list of texts, the list of tuples containing the best choices and their scores.

        Examples
        ---------
        >>> choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
        >>> p = Process()
        >>> p.extractOne("Cowboys", choices)
        ('Dallas Cowboys', 42.857142857142854)
        >>> p.extractOne(["Cowboys", "falcon's"], choices)
        [('Dallas Cowboys', 42.857142857142854), None]
        >>> p.extractOne(["Cowboys", "falcon's"], choices, score_cutoff=30)
        [('Dallas Cowboys', 42.857142857142854), ('Atlanta Falcons', 30.88235294117647)]
        """
        if isinstance(queries, str):
            is_str = True
            queries = [queries]
        else:
            is_str = False
        if choices is not None:
            self.fit(choices)
        jc_array = self.transform(queries)
        if is_str:
            return get_best_choice(self.choices, jc_array[0, :], score_cutoff)
        else:
            return [
                get_best_choice(self.choices, jc_array[i, :], score_cutoff)
                for i in range(len(queries))
            ]

    def dedupe(self, contains_dup, threshold=60.0):
        """
        Inspired by fuzzywuzzy's dedupe function to remove (near) duplicates.
        Currently barely optimized (and probably bugged).

        Parameters
        ----------
        contains_dup: :py:class:`list` of :py:class:`str`
            A list of texts with possible nearly redundant entries.
        threshold: :py:class:`float`, optional
            Minimal score that a result must achieve to be considered redundant.

        Returns
        -------
        :py:class:`list` of :py:class:`str`

        Examples
        --------
        >>> contains_dupes = ['Frodo Baggin', 'Frodo Baggins', 'F. Baggins', 'Samwise G.', 'Gandalf', 'Bilbo Baggins']
        >>> p = Process()
        >>> p.dedupe(contains_dupes)
        ['Frodo Baggins', 'F. Baggins', 'Samwise G.', 'Gandalf', 'Bilbo Baggins']

        `F. Baggins` is kept because the length difference impacts the results. Let us ignore the length.

        >>> p.length_impact = 0.0
        >>> p.dedupe(contains_dupes)
        ['Frodo Baggins', 'Samwise G.', 'Gandalf', 'Bilbo Baggins']
        """
        n = len(contains_dup)
        unprocessed = np.ones(n, dtype=bool)
        indexes = [None] * n
        x = self.vectorizer.fit_transform(contains_dup)
        y = x.T.tocsr()
        jc_matrix = jit_square_factors(
            x.indices, x.indptr, y.indices, y.indptr, n, self.length_impact
        )
        for i in range(n):
            if unprocessed[i]:
                closes = [j for j in range(n) if jc_matrix[i, j] > threshold]
                closest = closes[txt_max([contains_dup[i] for i in closes])]
                if indexes[closest] is not None:
                    closest = indexes[closest]
                for i in closes:
                    indexes[i] = closest
        return [contains_dup[i] for i in np.unique(indexes)]


# Trigger jit compilation on import.
def dry_run():
    p = Process()
    p.fit(["riri", "fifi", "rififi"])
    p.transform(["rir", "fido"])
    p.dedupe(["riri", "fifi", "rififi"])


dry_run()
