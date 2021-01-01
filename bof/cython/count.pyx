# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: extra_compile_args = -std=c++11

import cython
from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"

def make_rg(float sampling_rate=.5, int seed=42):
    cdef:
        mt19937 gen = mt19937(seed)
        uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
    def r():
        return dist(gen)<sampling_rate
    return r

cdef number_of_factors(int length, int n_range):
    """
    Return the number of factors (with multiplicity) of size at most `n_range` that exist in a text of length `length`.
    This allows to pre-allocate working memory.

    Parameters
    ----------
    length: :py:class:`int`
        Length of the text.
    n_range: :py:class:`int`
        Maximal factor size. If 0, all factors are considered.

    Returns
    -------
    int
        The number of factors (with multiplicity).

    Examples
    --------
    >>> l = len("riri")
    >>> number_of_factors(l)
    10
    >>> number_of_factors(l, n_range=2)
    7
    """
    if n_range == 0  or n_range > length:
        return length * (length + 1) // 2
    return n_range * (length - n_range) + n_range * (n_range + 1) // 2


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
@cython.nonecheck(False)
def fit_transform(list corpus, dict features, preprocessor,
                  int n_range=7):
    cdef str txt, sub_text, factor
    cdef int ptr=0, end, start, current, length, i, j, m=len(features), tot_size
    tot_size = sum(number_of_factors(len(preprocessor(txt)), n_range) for txt in corpus)
    cdef np.uint_t[::1] feature_indices
    feature_indices = np.zeros(tot_size, dtype=np.uint)
    cdef np.uint_t[::1] document_indices
    document_indices = np.zeros(tot_size, dtype=np.uint)
    for i, txt in enumerate(corpus):
        start_ptr = ptr
        txt = preprocessor(txt)
        length = len(txt)
        for start in range(length):
            end = min(start+n_range, length) if n_range>0 else length
            sub_text = txt[start:end]
            for current in range(1, end-start+1):
                factor = sub_text[:current]
                j = features.setdefault(factor, m)
                if j == m:
                    m += 1
                feature_indices[ptr] = j
                ptr += 1
        document_indices[start_ptr:ptr] = i

    return coo_matrix((np.ones(tot_size, dtype=np.uintc), (document_indices, feature_indices)),
                      shape=(len(corpus), m)).tocsr()

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
@cython.nonecheck(False)
def fit(list corpus, dict features, preprocessor,
                  int n_range=7):
    cdef str txt, sub_text, factor
    cdef int start, current, end, length, i, j, m=len(features)
    for txt in corpus:
        txt = preprocessor(txt)
        length = len(txt)
        for start in range(length):
            end = min(start+n_range, length) if n_range>0 else length
            sub_text = txt[start:end]
            for current in range(1, end-start+1):
                factor = sub_text[:current]
                j = features.setdefault(factor, m)
                if j == m:
                    m += 1


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
@cython.nonecheck(False)
def sampling_fit(list corpus, dict features, preprocessor,
                  int n_range=7, float sampling_rate=.5, int seed=42):

    rg = make_rg(sampling_rate, seed)

    cdef str txt, sub_text, factor
    cdef int start, current, end, length, i, j, m=len(features)
    for txt in corpus:
        txt = preprocessor(txt)
        length = len(txt)
        for start in range(length):
            if rg():
                end = min(start+n_range, length) if n_range>0 else length
                sub_text = txt[start:end]
                for current in range(1, end-start+1):
                    factor = sub_text[:current]
                    j = features.setdefault(factor, m)
                    if j == m:
                        m += 1


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
@cython.nonecheck(False)
def transform(list corpus, dict features, preprocessor,
                  int n_range=7):
    cdef str txt, sub_text, factor
    cdef int ptr=0, end, start, current, length, i, j, m=len(features), tot_size
    tot_size = sum(number_of_factors(len(preprocessor(txt)), n_range) for txt in corpus)
    cdef np.uint_t[::1] feature_indices
    feature_indices = np.zeros(tot_size, dtype=np.uint)
    cdef np.uint_t[::1] document_indices
    document_indices = np.zeros(tot_size, dtype=np.uint)
    for i, txt in enumerate(corpus):
        start_ptr = ptr
        txt = preprocessor(txt)
        length = len(txt)
        for start in range(length):
            end = min(start+n_range, length) if n_range>0 else length
            sub_text = txt[start:end]
            for current in range(1, end-start+1):
                factor = sub_text[:current]
                if factor in features:
                    feature_indices[ptr] = features[factor]
                    ptr += 1
        document_indices[start_ptr:ptr] = i

    feature_indices = feature_indices[:ptr]
    document_indices = document_indices[:ptr]

    return coo_matrix((np.ones(ptr, dtype=np.uintc), (document_indices, feature_indices)),
                      shape=(len(corpus), len(features))).tocsr()
