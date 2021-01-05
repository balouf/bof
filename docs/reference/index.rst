=========
Reference
=========


Fuzz
--------

The `fuzz` module mimicks the fuzzywuzzy-like package like

- fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy)
- rapidfuzz (https://github.com/maxbachmann/rapidfuzz)

The main difference is that the Levenshtein distance is replaced by the Joint Complexity distance. The API is also
slightly change to enable new features:

- The list of possible choices can be pre-trained (*fit*) to accelerate the computation in the case a stream of queries
  is sent against the same list of choices.
- Instead of one single query, a list of queries can be used. Computations will be parallelized.

The main `fuzz` entry point is the `Process` class.

.. automodule:: bof.fuzz
    :members:


Feature Extraction
-------------------

The `feature_extraction` module mimicks the module https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text
with a focus on character-based extraction.

The main differences are:

- it is slightly faster;
- the features can be incrementally updated;
- it is possible to fit only a random sample of factors to reduce space and computation time.

The main entry point for this module is the `CountVectorizer` class, which mimicks its *scikit-learn* counterpart.
It is in fact very similar to using `char` or `char_wb` analyzer option from that module.

.. automodule:: bof.feature_extraction
    :members:



Common
--------------------
The `common` module contains miscellaneous classes and functions.

.. automodule:: bof.common
    :members:
