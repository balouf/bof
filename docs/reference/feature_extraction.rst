The :py:mod:`~bof.feature_extraction` module mimics the module https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text
with a focus on character-based extraction.

The main differences are:

- it is slightly faster;
- the features can be incrementally updated;
- it is possible to fit only a random sample of factors to reduce space and computation time.

The main entry point for this module is the :py:class:`~bof.feature_extraction.CountVectorizer` class, which mimics
its *scikit-learn* counterpart (also named :class:`~sklearn.feature_extraction.text.CountVectorizer`).
It is in fact very similar to sklearn's :class:`~sklearn.feature_extraction.text.CountVectorizer` using `char` or
`char_wb` analyzer option from that module.
