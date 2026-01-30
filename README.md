# Bag of Factors

[![PyPI Status](https://img.shields.io/pypi/v/bof.svg)](https://pypi.python.org/pypi/bof)
[![Build Status](https://github.com/balouf/bof/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/balouf/bof/actions?query=workflow%3Abuild)
[![Documentation Status](https://github.com/balouf/bof/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/balouf/bof/actions?query=workflow%3Adocs)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://codecov.io/gh/balouf/bof/branch/master/graphs/badge.svg)](https://codecov.io/gh/balouf/bof/tree/main)


Bag of Factors allows you to analyze a corpus from its factors.


* Free software: MIT
* Documentation: https://balouf.github.io/bof/.


## Features


### Feature Extraction

The `feature_extraction` module mimics the module https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text
with a focus on character-based extraction.

The main differences are:

- it is slightly faster;
- the features can be incrementally updated;
- it is possible to fit only a random sample of factors to reduce space and computation time.

The main entry point for this module is the `CountVectorizer` class, which mimics
its *scikit-learn* counterpart (also named `CountVectorizer`).
It is in fact very similar to sklearn's `CountVectorizer` using `char` or
`char_wb` analyzer option from that module.


### Fuzz

The `fuzz` module mimics the fuzzywuzzy-like packages like

- fuzzywuzzy (https://github.com/seatgeek/fuzzywuzzy)
- rapidfuzz (https://github.com/maxbachmann/rapidfuzz)

The main difference is that the Levenshtein distance is replaced by the Joint Complexity distance. The API is also
slightly changed to enable new features:

- The list of possible choices can be pre-trained (`fit`) to accelerate the computation in
  the case a stream of queries is sent against the same list of choices.
- Instead of one single query, a list of queries can be used. Computations will be parallelized.

The main `fuzz` entry point is the `Process` class.


## Getting Started

Look at examples from the [reference] section.


## Credits

This package was created with [Cookiecutter] and the [package_helper_2] project template.

[Cookiecutter]: https://github.com/audreyr/cookiecutter
[package_helper_2]: https://github.com/francois-durand/package_helper_2
[reference]: https://balouf.github.io/bof/reference/index.html
