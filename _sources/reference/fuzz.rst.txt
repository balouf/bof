The :py:mod:`~bof.fuzz` module mimics the fuzzywuzzy-like packages like

- fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy)
- rapidfuzz (https://github.com/maxbachmann/rapidfuzz)

The main difference is that the Levenshtein distance is replaced by the Joint Complexity distance. The API is also
slightly changed to enable new features:

- The list of possible choices can be pre-trained (:py:meth:`~bof.fuzz.Process.fit`) to accelerate the computation in
  the case a stream of queries is sent against the same list of choices.
- Instead of one single query, a list of queries can be used. Computations will be parallelized.

The main :py:mod:`~bof.fuzz` entry point is the :py:class:`~bof.fuzz.Process` class.
