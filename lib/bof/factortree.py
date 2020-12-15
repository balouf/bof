from .common import MixInIO, default_preprocessor


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
        Keep for each factor a dict that tells for each document (represented by its index) the number of occurences of the factor in the document.
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
            if corpus is not None:
                self.fit_transform(corpus)

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
        >>> corpus = ["riri", "fifi", "rififi"]
        >>> tree = FactorTree(corpus=corpus, n_range=3)
        >>> tree.n
        3
        >>> tree.m
        13
        >>> tree.clear()
        >>> tree.n
        0
        >>> tree.m
        1
        >>> tree = FactorTree(corpus=corpus, n_range=3)
        >>> tree.clear(keep_graph=True)
        >>> tree.n
        0
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
        corpus: :py:class:`list` of :py:class:`list`.
            Texts to analyze.
        reset: :py:class:`bool`
            Clears FactorTree. If False, FactorTree will be updated instead.

        Returns
        -------
        None

        Examples
        --------
        >>> tree = FactorTree(n_range=3)
        >>> tree.fit_transform(["riri", "fifi", "rififi"])
        >>> tree.corpus
        ['riri', 'fifi', 'rififi']
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        >>> tree.fit_transform(["riri", "fifi"])
        >>> tree.corpus
        ['riri', 'fifi']
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi']
        >>> tree.fit_transform(["rififi"], reset=False)
        >>> tree.corpus
        ['riri', 'fifi', 'rififi']
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
        corpus: :py:class:`list` of :py:class:`list`.
            Texts to analyze.
        reset: :py:class:`bool`
            Clears FactorTree. If False, FactorTree will be updated instead.

        Returns
        -------
        None

        Examples
        --------
        >>> tree = FactorTree(n_range=3)
        >>> tree.fit(["riri", "fifi", "rififi"])
        >>> tree.corpus
        []
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        >>> tree.fit(["riri", "fifi"])
        >>> tree.corpus
        []
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi']
        >>> tree.fit(["rififi"], reset=False)
        >>> tree.corpus
        []
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        """
        if reset:
            self.clear()
        for txt in corpus:
            self.txt_fit(txt)


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
        >>> tree = FactorTree(n_range=3)
        >>> tree.fit_transform(["riri", "fifi", "rififi"])
        >>> tree.corpus
        ['riri', 'fifi', 'rififi']
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        >>> tree.count
        [{0: 5, 1: 5, 2: 7}, {0: 2, 2: 1}, {0: 2, 2: 1}, {0: 1}, {0: 2, 1: 2, 2: 3}, {0: 1}, {0: 1}, {1: 2, 2: 2}, {1: 2, 2: 2}, {1: 1, 2: 1}, {1: 1, 2: 2}, {1: 1, 2: 2}, {2: 1}]
        >>> tree.transform(["fir", "rfi"])
        >>> tree.corpus
        ['fir', 'rfi']
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        >>> tree.count
        [{0: 4, 1: 4}, {0: 1, 1: 1}, {}, {}, {0: 1, 1: 1}, {0: 1}, {}, {0: 1, 1: 1}, {0: 1, 1: 1}, {}, {}, {}, {}]
        >>> tree.transform(["rififi"], reset=False)
        >>> tree.corpus
        ['fir', 'rfi', 'rififi']
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        >>> tree.count
        [{0: 4, 1: 4, 2: 7}, {0: 1, 1: 1, 2: 1}, {2: 1}, {}, {0: 1, 1: 1, 2: 3}, {0: 1}, {}, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 2}, {2: 1}, {2: 2}, {2: 2}, {2: 1}]
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

        >>> tree = FactorTree()
        >>> tree.features
        ['']

        >>> tree.txt_fit_transform("riri")
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri']

        >>> tree.txt_fit_transform("rififi")
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'rif', 'rifi', 'rifif', 'if', 'ifi', 'ifif', 'ififi', 'f', 'fi', 'fif', 'fifi']
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

        >>> tree = FactorTree()
        >>> tree.features
        ['']

        >>> tree.txt_fit_transform("riri")
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri']

        >>> tree.txt_fit_transform("rififi")
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'rif', 'rifi', 'rifif', 'if', 'ifi', 'ifif', 'ififi', 'f', 'fi', 'fif', 'fifi']
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

        >>> tree = FactorTree()
        >>> tree.features
        ['']

        >>> tree.txt_fit_transform("riri")
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri']

        >>> tree.txt_fit_transform("rififi")
        >>> tree.features
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri', 'rif', 'rifi', 'rifif', 'if', 'ifi', 'ifif', 'ififi', 'f', 'fi', 'fif', 'fifi']
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

