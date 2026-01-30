# History

## 0.4.2 (2026-01-30): Compatibility

* Added Python 3.14 support.
* Minimum Python version bumped to 3.11.
* Dependencies updated: numba>=0.63.0, scipy>=1.16.0.
* Removed explicit numpy dependency (implicitly provided by numba and scipy).
* Fixed numba 'reflected list' deprecation warning by converting corpus to typed lists.

## 0.4.1 (2025-06-25): Maintenance

* Dependencies updated.
* Switched management to UV and documentation to Myst's MD.
* Ruff'd

## 0.4 (2021-04-08): Back to Numba

* Cython is too difficult to maintain and Numba dict management is relatively OK since last time. Time to switch!


## 0.3.5 (2021-04-08): ARM64

* Attempt to update PyPi with Mac M1 compatible wheels.


## 0.3.4 (2021-01-05): Cleaning

* Renaming process.py to fuzz.py to emphasize that the module aims at being an alternative to the fuzzywuzzy package.
* Removed modules FactorTree and JC. What they did is now essentially covered by the feature_extraction and fuzz
  modules.
* General cleaning / rewriting of the documentation.


## 0.3.3 (2021-01-01): Cython/Numba balanced

* All core CountVectorizer methods ported to Cython. Roughly 2.5X faster than sklearn counterpart (mainly because some features like min_df/max_df are not implemented).
* Process numba methods NOT converted to Cython as Numba seems to be 20% faster for csr manipulation.
* Numba functions are cached to avoid compilation lag.


## 0.3.2 (2020-12-30): Going Cython

* First attempt to use Cython
* Right now only the fit_transform method of CountVectorizer has been cythonized, for testing wheels.
* If all goes well, numba will probably be abandoned and all the heavy-lifting will be in Cython.


## 0.3.1 (2020-12-28): Simplification of core algorithm

* Attributes of the CountVectorizer have been reduced to the minimum: one dict!
* Now faster than sklearn counterpart! (The reason been only one case is considered here so we can ditch a lot of checks and attributes).


## 0.3.0 (2020-12-15): CountVectorizer and Process

* The core is now the CountVectorizer class. Lighter and faster. Only features are kept inside.
* New process module inspired by fuzzywuzzy!


## 0.2.0 (2020-12-15): Fit/Transform

* Full refactoring to make the package fit/transform compliant.
* Add a fit_sampling method that allows to fit only a (random) subset of factors


## 0.1.1 (2020-12-12): Upgrades

* Docstrings added
* Common module (feat. save/load capabilities)
* Joint Complexity module

## 0.1.0 (2020-12-12): First release

* First release on PyPI.
* Core FactorTree class added.
