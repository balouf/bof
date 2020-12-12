from .common import MixInIO, default_preprocessor


class FactorTree(MixInIO):
    """
    Maintain a tree of factor of a given corpus.

    Parameters
    ----------
    corpus: :py:class:`list` of :py:class:`str`, optional
        Corpus of documents to decompose into factors.
    auto_update: :py:class:`bool`, optional
        When processing external texts, should the tree be updated (TO BE MOVED WITH THE FACTORS MODULE WHEN CREATED).
    preprocessor: callable
        Preprocessing function to apply to texts before adding them to the factor tree.
    n_range: :py:class:`int` or None, optional
        Maximum factor size. If `None`, all factors will be extracted.

    Attributes
    ----------
    count: :py:class:`list` of :py:class:`dict`
        Keep for each factor a dict that tells for each document (represented by its index) the number of occurences of the factor in the document.
    edges: :py:class:`list` of :py:class:`dict`
        Keep for each factor a dict that associates to each letter the corresponding factor index in the tree (if any).
    corpus_list: :py:class:`list` of :py:class:`srt`
        The corpus list.
    corpus_dict: :py:class:`dict` of :py:class:`str` -> :py:class:`int`
        Reverse index of the corpus (`corpus_dict[corpus_list[i]] == i`).
    factor_list: :py:class:`list` of :py:class:`srt`
        The factor list.
    factor_dict: :py:class:`dict` of :py:class:`str` -> :py:class:`int`
        Reverse index of the factors (`factor_dict[factor_list[i]] == i`).
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

    >>> tree.factor_list
    ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
    """
    def __init__(self, corpus=None, auto_update=False, preprocessor=None, n_range=5, filename=None, path='.'):
        if filename is not None:
            self.load(filename=filename, path=path)
        else:
            self.count = [dict()]
            self.edges = [dict()]
            self.corpus_list = []
            self.corpus_dict = dict()
            self.factor_list = [""]
            self.factor_dict = {"": 0}
            self.self_factors = []
            self.m = 1
            self.n = 0
            self.auto_update = auto_update
            if preprocessor is None:
                preprocessor = default_preprocessor
            self.preprocessor = preprocessor
            self.n_range = n_range
            if corpus is not None:
                self.add_txt_list_to_tree(corpus)

    def add_txt_list_to_tree(self, txt_list):
        """
        Add a list of texts to the factor tree.

        Parameters
        ----------
        txt_list: :py:class:`list` of :py:class:`srt`
            Texts to add.

        Returns
        -------
        None

        Examples
        ---------

        >>> tree = FactorTree(n_range=3)
        >>> tree.add_txt_list_to_tree(["riri", "fifi"])
        >>> tree.factor_list
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi']

        >>> tree.add_txt_list_to_tree(["rififi"])
        >>> tree.factor_list
        ['', 'r', 'ri', 'rir', 'i', 'ir', 'iri', 'f', 'fi', 'fif', 'if', 'ifi', 'rif']
        """
        for txt in txt_list:
            self.add_txt_to_tree(txt)

    def add_txt_to_tree(self, txt):
        """
        Add a text to the factor tree.

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
        >>> tree.factor_list
        ['']

        >>> tree.add_txt_to_tree("riri")
        >>> tree.factor_list
        ['', 'r', 'ri', 'rir', 'riri', 'i', 'ir', 'iri']

        >>> tree.add_txt_to_tree("rififi")
        >>> tree.factor_list
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
                n_node = self.edges[node].setdefault(letter, self.m)
                if n_node == self.m:
                    self.edges.append(dict())
                    self.count.append(dict())
                    fac = self.factor_list[node] + letter
                    self.factor_list.append(fac)
                    self.factor_dict[fac] = self.m
                    self.m += 1
                node = n_node
                d = self.count[node]
                if d.setdefault(self.n, 0) == 0:
                    self.self_factors[self.n] += 1
                d[self.n] += 1
        self.corpus_list.append(txt)
        self.corpus_dict[txt] = self.n
        self.n += 1
