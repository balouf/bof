=======
History
=======

---------------------------------------------------
0.3.3 (2021-01-01): Cython/Numba balanced
---------------------------------------------------

* All core CountVectorizer methods ported to Cython. Roughly 2.5X faster than sklearn counterpart (mainly because some features like min_df/max_df are not implemented).
* Process numba methods NOT converted to Cython as Numba seems to be 20% faster for csr manipulation.
* Numba functions are cached to avoid compilation lag.


---------------------------------------------------
0.3.2 (2020-12-30): Going Cython
---------------------------------------------------

* First attempt to use Cython
* Right now only the fit_transform method of CountVectorizer has been cythonized, for testing wheels.
* If all goes well, numba will probably be abandoned and all the heavy-lifting will be in Cython.


-----------------------------------------------------
0.3.1 (2020-12-28): Simplification of core algorithm
-----------------------------------------------------

* Attributes of the CountVectorizer have been reduced to the minimum: one dict!
* Now faster than sklearn counterpart! (The reason been only one case is considered here so we can ditch a lot of checks and attributes).


---------------------------------------------------
0.3.0 (2020-12-15): CountVectorizer and Process
---------------------------------------------------

* The core is now the CountVectorizer class. Lighter and faster. Only features are kept inside.
* New process module inspired by fuzzywuzzy!


---------------------------------
0.2.0 (2020-12-15): Fit/Transform
---------------------------------

* Full refactoring to make the package fit/transform compliant.
* Add a fit_sampling method that allows to fit only a (random) subset of factors


---------------------------------
0.1.1 (2020-12-12): Upgrades
---------------------------------

* Docstrings added
* Common module (feat. save/load capabilities)
* Joint Complexity module

---------------------------------
0.1.0 (2020-12-12): First release
---------------------------------

* First release on PyPI.
* Core FactorTree class added.
