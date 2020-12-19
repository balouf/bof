from .common import MixInIO, default_preprocessor, make_random_bool_generator, set_seed


class FactorTree(MixInIO):
    """
    Maintain a tree of factor of a given corpus.

    Parameters
    ----------
    corpus: :py:class:`list` of :py:class:`str`, optional
        Corpus of documents to decompose into factors.
    preprocessor: callable
        Preprocessing function to apply to texts before adding them to the factor tree.
    n_range: :py:class:`int` or None, optional
        Maximum factor size. If `None`, all factors will be extracted.

    Attributes
    ----------
    count: :py:class:`list` of :py:class:`dict`
        Keep for each factor a dict that tells for each document (represented by its index) the number of occurences of
        the factor in the document.
    graph: :py:class:`list` of :py:class:`dict`
        Keep for each factor a dict that associates to each letter the corresponding factor index in the tree (if any).
    corpus: :py:class:`list` of :py:class:`srt`
        The corpus list.
    corpus_: :py:class:`dict` of :py:class:`str` -> :py:class:`int`
        Reverse index of the corpus (`corpus_[corpus_list[i]] == i`).
    features: :py:class:`list` of :py:class:`srt`
        The factor list.
    features_: :py:class:`dict` of :py:class:`str` -> :py:class:`int`
        Reverse index of the factors (`features_[factor_list[i]] == i`).
    self_factors: :py:class:`list` of :py:class:`int`
        Number of unique factors for each text.
    n: :py:class:`int`
        Number of texts.
    m: :py:class:`int`
        Number of factors.



    Examples
    --------

    Build a tree from a corpus of texts,limiting factor size to 3:

    >>> corpus = ["riri", "fifi", "rififi"]
    >>> tree = FactorTree(corpus=corpus, n_range=3)

    List the number of unique factors for each text:

    >>> tree.self_factors
    [7, 7, 10]

    List the factors in the corpus:

    >>> tree.features
    ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

    Display the factors per document:

    >>> print(count_display(tree))
    riri: ''x5, 'r'x2, 'ri'x2, 'rir'x1, 'i'x2, 'ir'x1, 'iri'x1
    fifi: ''x5, 'i'x2, 'f'x2, 'fi'x2, 'fif'x1, 'if'x1, 'ifi'x1
    rififi: ''x7, 'r'x1, 'ri'x1, 'i'x3, 'f'x2, 'fi'x2, 'fif'x1, 'if'x2, 'ifi'x2, 'rif'x1
    """
    def __init__(self, corpus=None, preprocessor=None, n_range=5, filename=None, path='.'):
        if filename is not None:
            self.load(filename=filename, path=path)
        else:
            self.count = [dict()]
            self.graph = [dict()]
            self.corpus = []
            self.corpus_ = dict()
            self.features = [""]
            self.features_ = {"": 0}
            self.self_factors = []
            self.m = 1
            self.n = 0
            if preprocessor is None:
                preprocessor = default_preprocessor
            self.preprocessor = preprocessor
            self.n_range = n_range
            self.rb = None
            if corpus is not None:
                self.fit_transform(corpus)

    def set_rb(self, sampling_rate=.5):
        self.rb = make_random_bool_generator(probability_true=sampling_rate)

    def clear(self, keep_graph=False):
        """
        Reset the object.

        Parameters
        ----------
        keep_graph: :py:class:`bool`, optional
            Preserve the factor tree, only clear the corpus related part.

        Returns
        -------
        None

        Examples
        --------

        Build a factor tree of three documents, with max factor size 3.

        >>> corpus = ["riri", "fifi", "rififi"]
        >>> tree = FactorTree(corpus=corpus, n_range=3)

        Number of documents:

        >>> tree.n
        3

        Number of factors:

        >>> tree.m
        13

        Clear the object.

        >>> tree.clear()

        Number of documents:

        >>> tree.n
        0

        Number of factors (remind that the empty factor "" is always present).

        >>> tree.m
        1

        Rebuild the FactorTree.

        >>> tree = FactorTree(corpus=corpus, n_range=3)

        Clear the corpus part, leaving the factor graph untouched..

        >>> tree.clear(keep_graph=True)

        Number of documents:

        >>> tree.n
        0

        Number of factors:

        >>> tree.m
        13
        """
        self.corpus = []
        self.corpus_ = dict()
        self.self_factors = []
        self.n = 0
        if keep_graph:
            self.count = [dict() for _ in range(self.m)]
        else:
            self.count = [dict()]
            self.graph = [dict()]
            self.features = [""]
            self.features_ = {"": 0}
            self.m = 1

    def fit_transform(self, corpus, reset=True):
        """
        Build the factor tree and populate the factor counts.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`str`.
            Texts to analyze.
        reset: :py:class:`bool`
            Clears FactorTree. If False, FactorTree will be updated instead.

        Returns
        -------
        None

        Examples
        --------

        Build a FactorTree from a corpus of three documents:

        >>> tree = FactorTree(n_range=3)
        >>> tree.fit_transform(["riri", "fifi", "rififi"])

        List of documents:

        >>> tree.corpus
        ['riri', 'fifi', 'rififi']

        List of factors (of size at most 3):

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

        Build a FactorTree from a corpus of two documents.

        >>> tree.fit_transform(["riri", "fifi"])

        Notice the implicit reset, as there are now two documents:

        >>> tree.corpus
        ['riri', 'fifi']

        Similarly, the factors are these from ``riri`` and ``fifi``.

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi']

        With `reset` set to `False`, we can add another list while keeping the previous state.

        >>> tree.fit_transform(["rififi"], reset=False)

        We have now 2+1=3 documents.

        >>> tree.corpus
        ['riri', 'fifi', 'rififi']

        The list of features has been updated as well:

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        """
        if reset:
            self.clear()
        for txt in corpus:
            self.txt_fit_transform(txt)

    def fit(self, corpus, reset=True):
        """
        Build the factor tree. Does not update inner corpus.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`str`.
            Texts to analyze.
        reset: :py:class:`bool`
            Clears FactorTree. If False, FactorTree will be updated instead.

        Returns
        -------
        None

        Examples
        --------

        We fit a corpus to a tree.

        >>> tree = FactorTree(n_range=3)
        >>> tree.fit(["riri", "fifi", "rififi"])

        The inner corpus remains empty:

        >>> tree.corpus
        []

        The factors have been populated:

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']

        We fit another corpus.

        >>> tree.fit(["riri", "fifi"])

        The inner corpus remains empty:

        >>> tree.corpus
        []

        The factors have been implicitly reset and updated from the new corpus (`rif` is gone in this toy example):

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi']

        We add another corpus to the fit by setting `reset` to `False`:

        >>> tree.fit(["rififi"], reset=False)

        The inner corpus remains empty:

        >>> tree.corpus
        []

        The list of features has been updated (with `rif``):

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        """
        if reset:
            self.clear()
        for txt in corpus:
            self.txt_fit(txt)

    def sampling_fit(self, corpus, reset=True, sampling_rate=.5):
        """
        Build a partial factor tree where only a random subset of factors are selected. Does not update inner corpus.

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

        >>> tree = FactorTree()
        >>> tree.fit(["riri", "fifi", "rififi"])
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'fifi', 'if', 'ifi', 'rif', 'rifi', 'rifif', 'ifif', 'ififi']

        Now we use a sampling fit instead. Only a subset of the factors are selected.

        >>> set_seed(2012)
        >>> tree.sampling_fit(["riri", "fifi", "rififi"])
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi', 'if', 'ifi', 'ifif', 'ififi']

        We random fit another corpus. We reset the seed to reproduce the example above.

        >>> set_seed(2012)
        >>> tree.sampling_fit(["riri", "fifi"])

        The factors have been implicitly reset.

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi']

        We add another corpus to the fit by setting `reset` to `False`:

        >>> tree.sampling_fit(["rififi"], reset=False)

        The list of features has been updated:

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'f', 'fi', 'if', 'ifi', 'ifif', 'ififi']
        """
        if reset:
            self.clear()
        self.rb = make_random_bool_generator(probability_true=sampling_rate)
        for txt in corpus:
            self.txt_sampling_fit(txt)

    def transform(self, corpus, reset=True):
        """
        Counts factors from the factor tree in the corpus. Does not update the factor tree.

        Parameters
        ----------
        corpus: :py:class:`list` of :py:class:`list`.
            Texts to analyze.
        reset: :py:class:`bool`
            Clears internal corpus. If False, internal corpus will be updated instead.

        Returns
        -------
        None

        Examples
        --------

        To start, we fit_transform a corpus:

        >>> tree = FactorTree(n_range=3)
        >>> tree.fit_transform(["riri", "fifi", "rififi"])

        The corpus decomposition in factors:

        >>> print(count_display(tree))
        riri: ''x5, 'r'x2, 'ri'x2, 'rir'x1, 'i'x2, 'ir'x1, 'iri'x1
        fifi: ''x5, 'i'x2, 'f'x2, 'fi'x2, 'fif'x1, 'if'x1, 'ifi'x1
        rififi: ''x7, 'r'x1, 'ri'x1, 'i'x3, 'f'x2, 'fi'x2, 'fif'x1, 'if'x2, 'ifi'x2, 'rif'x1

        We now apply a transform.

        >>> tree.transform(["fir", "rfi"])

        Observe the corpus decomposition: the old corpus has been erased, and new factors (e.g. `rf`) are discarded.

        >>> print(count_display(tree))
        fir: ''x4, 'r'x1, 'i'x1, 'ir'x1, 'f'x1, 'fi'x1
        rfi: ''x4, 'r'x1, 'i'x1, 'f'x1, 'fi'x1

        We update (without discarding previous entries) the corpus with a new one.

        >>> tree.transform(["rififi"], reset=False)
        >>> print(count_display(tree))
        fir: ''x4, 'r'x1, 'i'x1, 'ir'x1, 'f'x1, 'fi'x1
        rfi: ''x4, 'r'x1, 'i'x1, 'f'x1, 'fi'x1
        rififi: ''x7, 'r'x1, 'ri'x1, 'i'x3, 'f'x2, 'fi'x2, 'fif'x1, 'if'x2, 'ifi'x2, 'rif'x1
        """
        if reset:
            self.clear(keep_graph=True)
        for txt in corpus:
            self.txt_transform(txt)

    def txt_fit_transform(self, txt):
        """
        Add a text to the factor tree and update count.

        Parameters
        ----------
        txt: :py:class:`srt`
            Text to add.

        Returns
        -------
        None

        Examples
        ---------

        Starts with empty corpus.

        >>> tree = FactorTree()
        >>> tree.features
        ['']

        Add one document. Check factors.

        >>> tree.txt_fit_transform("riri")
        >>> print(count_display(tree))
        riri: ''x5, 'r'x2, 'ri'x2, 'rir'x1, 'riri'x1, 'i'x2, 'ir'x1, 'iri'x1

        Add another document. Check factors.

        >>> tree.txt_fit_transform("rififi")
        >>> print(count_display(tree))
        riri: ''x5, 'r'x2, 'ri'x2, 'rir'x1, 'riri'x1, 'i'x2, 'ir'x1, 'iri'x1
        rififi: ''x7, 'r'x1, 'ri'x1, 'i'x3, 'rif'x1, 'rifi'x1, 'rifif'x1, 'if'x2, 'ifi'x2, 'ifif'x1, 'ififi'x1, 'f'x2, 'fi'x2, 'fif'x1, 'fifi'x1
        """

        txt = self.preprocessor(txt)
        length = len(txt)

        # Empty factor my friend!
        self.count[0][self.n] = len(txt) + 1
        self.self_factors.append(1)

        for start in range(length):
            node = 0
            end = min(start + self.n_range, length) if self.n_range else length
            for letter in txt[start:end]:
                n_node = self.graph[node].setdefault(letter, self.m)
                if n_node == self.m:
                    self.graph.append(dict())
                    self.count.append(dict())
                    fac = self.features[node] + letter
                    self.features.append(fac)
                    self.features_[fac] = self.m
                    self.m += 1
                node = n_node
                d = self.count[node]
                if d.setdefault(self.n, 0) == 0:
                    self.self_factors[self.n] += 1
                d[self.n] += 1
        self.corpus.append(txt)
        self.corpus_[txt] = self.n
        self.n += 1

    def txt_fit(self, txt):
        """
        Add a text's factor to the factor tree. Do not update count.

        Parameters
        ----------
        txt: :py:class:`srt`
            Text to update factors from.

        Returns
        -------
        None

        Examples
        ---------

        Initiate a factor tree.

        >>> tree = FactorTree()
        >>> tree.features
        ['']

        Fit one document. Check features.

        >>> tree.txt_fit("riri")
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri']

        Fit another document. Check features.

        >>> tree.txt_fit("rififi")
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'rif', 'rifi', 'rifif', 'if', 'ifi', 'ifif', 'ififi', 'f', 'fi', 'fif', 'fifi']

        The corpus stays empty.

        >>> tree.corpus
        []
        """

        txt = self.preprocessor(txt)
        length = len(txt)

        for start in range(length):
            node = 0
            end = min(start + self.n_range, length) if self.n_range else length
            for letter in txt[start:end]:
                n_node = self.graph[node].setdefault(letter, self.m)
                if n_node == self.m:
                    self.graph.append(dict())
                    fac = self.features[node] + letter
                    self.features.append(fac)
                    self.features_[fac] = self.m
                    self.m += 1
                node = n_node

    def txt_sampling_fit(self, txt):
        """
        Add a text's factor to the factor tree.
        When a new factor is met, it is added with some probability, otherwise it is discarded.
        This can be seen as a simple way to reduce the size of the tree by only selecting a subset of frequent factors.
        Do not update count.

        Parameters
        ----------
        txt: :py:class:`srt`
            Text to update factors from.

        Returns
        -------
        None

        Examples
        ---------

        Initiate tree

        >>> tree = FactorTree()

        Prepare random boolean generator.

        >>> tree.set_rb()

        Initiate seed
        >>> set_seed(42)

        Random fit a few words.

        >>> tree.txt_sampling_fit("riri")
        >>> tree.txt_sampling_fit("fifi")
        >>> tree.txt_sampling_fit("rififi")

        We can observe that only a (random) subset of factors have been selected. The list is coherent: it contains the
        subfactors of each factor.

        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'f', 'fi', 'fif', 'fifi', 'i', 'if', 'ifi']
        """

        txt = self.preprocessor(txt)
        length = len(txt)

        for start in range(length):
            if self.rb():
                node = 0
                end = min(start + self.n_range, length) if self.n_range else length
                for letter in txt[start:end]:
                    n_node = self.graph[node].setdefault(letter, self.m)
                    if n_node == self.m:
                        self.graph.append(dict())
                        fac = self.features[node] + letter
                        self.features.append(fac)
                        self.features_[fac] = self.m
                        self.m += 1
                    node = n_node

    def txt_transform(self, txt):
        """
        Parse a text through the factor tree without updating it. Update count.

        Parameters
        ----------
        txt: :py:class:`srt`
            Text to add.

        Returns
        -------
        None

        Examples
        ---------

        Initiate a factor tree.

        >>> tree = FactorTree()
        >>> tree.features
        ['']

        Add a document. Check factors.

        >>> tree.txt_fit_transform("riri")
        >>> print(count_display(tree))
        riri: ''x5, 'r'x2, 'ri'x2, 'rir'x1, 'riri'x1, 'i'x2, 'ir'x1, 'iri'x1

        transform another document. Check factors. Only pre-existing factors have been used for the second document.

        >>> tree.txt_transform("rififi")
        >>> print(count_display(tree))
        riri: ''x5, 'r'x2, 'ri'x2, 'rir'x1, 'riri'x1, 'i'x2, 'ir'x1, 'iri'x1
        rififi: ''x7, 'r'x1, 'ri'x1, 'i'x3
        """

        txt = self.preprocessor(txt)
        length = len(txt)

        # Empty factor my friend!
        self.count[0][self.n] = len(txt) + 1
        self.self_factors.append(1)

        for start in range(length):
            node = 0
            end = min(start + self.n_range, length) if self.n_range else length
            for letter in txt[start:end]:
                node = self.graph[node].get(letter)
                if node is None:
                    break
                d = self.count[node]
                if d.setdefault(self.n, 0) == 0:
                    self.self_factors[self.n] += 1
                d[self.n] += 1
        self.corpus.append(txt)
        self.corpus_[txt] = self.n
        self.n += 1


def count_display(tree):
    """
    Simple side function to check the factor decomposition on a small corpus. DO NOT USE ON LARGE CORPI!

    Parameters
    ----------
    tree: :class:`~bof.factortree.FactorTree`

    Returns
    -------
    str

    Examples
    --------

    Analyzes the factor of size at most 3 of `["riri", "fifi", "rififi"]`:

    >>> tree = FactorTree(["riri", "fifi", "rififi"], n_range=3)
    >>> print(count_display(tree))
    riri: ''x5, 'r'x2, 'ri'x2, 'rir'x1, 'i'x2, 'ir'x1, 'iri'x1
    fifi: ''x5, 'i'x2, 'f'x2, 'fi'x2, 'fif'x1, 'if'x1, 'ifi'x1
    rififi: ''x7, 'r'x1, 'ri'x1, 'i'x3, 'f'x2, 'fi'x2, 'fif'x1, 'if'x2, 'ifi'x2, 'rif'x1
    """
    def doc_factors(i):
        factors = ", ".join([f"'{tree.features[j]}'x{c_dict[i]}" for j, c_dict in enumerate(tree.count) if i in c_dict])
        return f"{tree.corpus[i]}: {factors}"

    return "\n".join(doc_factors(i) for i in range(tree.n))
