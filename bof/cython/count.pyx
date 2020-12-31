# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from libcpp cimport bool

import cython
from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function


cdef number_of_factors(int length, n_range=None):
    """Testing ds"""
    if n_range is None or n_range > length:
        return length * (length + 1) // 2
    return n_range * (length - n_range) + n_range * (n_range + 1) // 2


def c_fit_transform(list corpus, dict features_, preprocessor,
                  int n_range=7, bool use_range=True):
    cdef str txt, f, letter
    cdef int ptr=0, end, start, length
    tot_size = sum(number_of_factors(len(preprocessor(txt)), n_range) for txt in corpus)
    cdef np.ndarray[np.uint_t] feature_indices = np.zeros(tot_size, dtype=np.uint)
    cdef np.ndarray[np.uint_t] document_indices = np.zeros(tot_size, dtype=np.uint)
    for i, txt in enumerate(corpus):
        start_ptr = ptr
        txt = preprocessor(txt)
        length = len(txt)
        for start in range(length):
            f = ""
            end = min(start+n_range, length) if use_range else length
            for letter in txt[start:end]:
                f += letter
                feature_indices[ptr] = features_.setdefault(f, len(features_))
                ptr += 1 # next(ptr_iter)
        document_indices[start_ptr:ptr] = i
    m = len(features_)

    return coo_matrix((np.ones(tot_size, dtype=np.uintc), (document_indices, feature_indices)),
                      shape=(len(corpus), m)).tocsr()
