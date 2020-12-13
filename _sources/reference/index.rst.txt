=========
Reference
=========

CountVectorizer
----------------

The `CountVectorizer` class mimicks the module https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html. It is in fact very similar to using `char` or `char_wb` analyzer option from that module.

TO BE WRITTEN?

Fuzz
------

The `fuzz` modules mimicks the fuzz module from

- fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy)
- rapidfuzz (https://github.com/maxbachmann/rapidfuzz)

The main difference is that the sets of available scorers differ, as the Levenshtein distance is replaced by the Joint Complexity distance.

MODULE TO BE WRITTEN


Fuzz
------

The `fuzz` modules mimicks the fuzz module from

- fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy)
- rapidfuzz (https://github.com/maxbachmann/rapidfuzz)

The main difference is that the sets of available scorers differ, as the Levenshtein distance is replaced by the Joint Complexity distance.

MODULE TO BE WRITTEN

Process
--------

The `process` modules mimicks the process module from

- fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy)
- rapidfuzz (https://github.com/maxbachmann/rapidfuzz)

The main difference is that the sets of available scorers differ, as the Levenshtein distance is replaced by the Joint Complexity distance.

MODULE TO BE WRITTEN

Factor Tree
--------------------

The `factortree` module contains the core engine of the *Bag of Factors* package.

.. automodule:: bof.factortree
    :members:


Joint Complexity
--------------------

The `jc` module contains function to compute joint complexity between texts.

.. automodule:: bof.jc
    :members:


Common
--------------------
The `common` module contains miscellaneous classes and functions.

.. automodule:: bof.common
    :members:
