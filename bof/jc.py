from.factortree import FactorTree


def common_factors(txt, tree, update_tree=False):
    """
    Compute the number of common factors between a text and a tree corpus.

    - Checks if the text belongs to the corpus first;
    - Update the tree if `update_tree` is True.

    Parameters
    ----------
    txt: :py:class:`str`
        Text to analyze
    tree: :class:`~bof.factortree.FactorTree`
        Factor tree of the corpus.
    update_tree: :py:class:`bool`
        Update tree if `txt` is not in it?

    Returns
    -------
    factors: :py:class:`list` of :py:class:`int`
        List of numbers of unique common factors.
    self_factor: :py:class:`int`
        Number of unique factors of the reference text.

    Examples
    --------

    >>> tree = FactorTree(["riri", "fifi", "rififi"])
    >>> common_factors("riri", tree)
    ([8, 2, 4], 8)
    >>> common_factors("rafi", tree)
    ([3, 4, 5], 11)
    >>> tree.corpus_list
    ['riri', 'fifi', 'rififi']
    >>> common_factors("fifi", tree, update_tree=True)
    ([2, 8, 8], 8)
    >>> tree.corpus_list
    ['riri', 'fifi', 'rififi']
    >>> common_factors("fari", tree, update_tree=True)
    ([4, 3, 5, 11], 11)
    >>> tree.corpus_list
    ['riri', 'fifi', 'rififi', 'fari']
    """
    txt = tree.preprocessor(txt)
    index = tree.corpus_dict.get(txt)
    if not index:
        if not update_tree:
            return common_factors_external(txt, tree)
        index = tree.n
        tree.add_txt_to_tree(txt)
    return common_factors_internal(index, tree)


def common_factors_internal(i, tree):
    """
    Compute the number of common factors between text of index `i` and the tree corpus.

    Parameters
    ----------
    i: :py:class:`int`
        Index of the reference text
    tree: :class:`~bof.factortree.FactorTree`
        Factor tree of the corpus.

    Returns
    -------
    factors: :py:class:`list` of :py:class:`int`
        List of numbers of unique common factors.
    self_factor: :py:class:`int`
        Number of unique factors of the reference text.

    Examples
    --------

    >>> tree = FactorTree(["riri", "fifi", "rififi"])
    >>> common_factors_internal(0, tree)
    ([8, 2, 4], 8)
    >>> common_factors_internal(1, tree)
    ([2, 8, 8], 8)
    >>> common_factors_internal(2, tree)
    ([4, 8, 15], 15)
    """
    buffer = {0}
    factors = [0] * tree.n
    while buffer:
        node = buffer.pop()
        if i in tree.count[node]:
            for j in tree.count[node]:
                factors[j] += 1
            buffer.update(tree.edges[node].values())
    return factors, tree.self_factors[i]


def common_factors_external(txt, tree):
    """
    Compute the number of common factors between a text and a tree corpus.

    - Does not check if the text belongs to the corpus;
    - Does not modify the tree.

    Parameters
    ----------
    txt: :py:class:`str`
        Text to analyze
    tree: :class:`~bof.factortree.FactorTree`
        Factor tree of the corpus.

    Returns
    -------
    factors: :py:class:`list` of :py:class:`int`
        List of numbers of unique common factors.
    self_factor: :py:class:`int`
        Number of unique factors of the reference text.

    Examples
    --------

    >>> tree = FactorTree(["riri", "fifi", "rififi"])
    >>> common_factors_external("riri", tree)
    ([8, 2, 4], 8)
    >>> common_factors_external("rafi", tree)
    ([3, 4, 5], 11)
    >>> tree.corpus_list
    ['riri', 'fifi', 'rififi']
    """

    buffer = {0}
    factors = [0] * tree.n
    new_tree = FactorTree([txt], preprocessor=tree.preprocessor, n_range=tree.n_range)
    while buffer:
        node = buffer.pop()
        target = tree.factor_dict.get(new_tree.factor_list[node])
        if target is not None:
            for j in tree.count[target]:
                factors[j] += 1
            buffer.update(new_tree.edges[node].values())
    return factors, new_tree.self_factors[0]


def joint_complexity(txt, tree, balance=.5, update_tree=False):
    """
    Computes Joint Complexity between a text and a corpus.

    Parameters
    ----------
    txt: :py:class:`str`
        Text to analyze
    tree: :class:`~bof.factortree.FactorTree`
        Factor tree of the corpus.
    balance: :py:class:`float`in range [0.0, 1.0]
        Large values will disregard the size of corpus elements..
    update_tree: :py:class:`bool`
        Update tree if `txt` is not in it?

    Returns
    -------
    :py:class:`list` of :py:class:`float`
        The Joint Complexity distances between the text and corpus.

    Examples
    --------

    >>> tree = FactorTree(["riri", "fifi", "rififi"], n_range=None)
    >>> joint_complexity("fififi", tree)
    [0.8888888888888888, 0.3333333333333333, 0.4444444444444444]
    >>> joint_complexity("rifi", tree)
    [0.7142857142857143, 0.5, 0.375]

    With low balance, `rifi` is closer to `fifi` than `rififi` because `rififi` is longer.

    >>> joint_complexity("rifi", tree, balance=0.0)
    [0.6666666666666666, 0.4, 0.5454545454545454]

    With high bbalance, `rifi` is closer to `rififi` than `fifi` because `rififi` may be longer but it contains `rifi`.

    >>> joint_complexity("rifi", tree, balance=1.0)
    [0.75, 0.5714285714285714, 0.0]
    """
    cofactors, self_factors = common_factors(txt, tree, update_tree=update_tree)
    biased_factors = [2 * balance * self_factors + 2 * (1 - balance) * f for f in tree.self_factors]
    return [0 if biased_factor - cofactor == 0 else
            (biased_factor - 2 * cofactor) /
            (biased_factor - cofactor)
            for biased_factor, cofactor in zip(biased_factors, cofactors)]
