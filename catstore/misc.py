import numpy as np


# http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float-in-python
def is_number(s):
    try:
        float(s)
        return True
    except TypeError:
        return False
    except ValueError:
        return False


# as seen in hickle (https://github.com/telegraphic/hickle/blob/master/hickle.py)
def check_is_iterable(py_obj):
    """ Check whether a python object is iterable.
    Note: this treats unicode and string as NON ITERABLE
    Args:
        py_obj: python object to test
    Returns:
        iter_ok (bool): True if item is iterable, False is item is not
    """
    if type(py_obj) in (str, str):
        return False
    try:
        iter(py_obj)
        return True
    except TypeError:
        return False


def dict_reverse_lookup(d, v):
    """ Find the key corresponding to the value in a dictionary.
    See http://stackoverflow.com/a/2569076

    """
    try:
        key = next((key for key, value in list(d.items()) if value == v))
    except StopIteration:
        raise ValueError("Item not found in dictionary (%s)"%str(v))
    return key


def dict_to_structured_array(data_dict):
    """ Convert a dictionary of numpy arrays to a structured array.

    Inputs
    ------
    data_dict - dictionary of numpy arrays

    Outputs
    -------
    structured array
    """
    lengths = []
    dtypes = []
    for name, arr in list(data_dict.items()):
        lengths.append(arr.shape[0])
        dim = 1
        if len(arr.shape) > 1:
            dim = arr.shape[1:]
        dtypes.append((name, arr.dtype, dim))

    lengths = np.array(lengths)
    if not np.all(lengths == lengths[0]):
        raise ValueError("Not all arrays in the dictionary have the same length.")

    # initialize the empty structured array
    struc_array = np.zeros(lengths[0], dtype=dtypes)

    # load the data
    for name, arr in list(data_dict.items()):
        struc_array[name] = arr

    return struc_array


def concatenate_dtypes(dtypes):
    """ Combine a list of dtypes.
    This may be used to add columns to a dtype.

    Parameters
    ----------
    dtypes : sequence
        sequence of dtypes

    Results
    ---------
    dtype : numpy.dtype
        the combined dtype object
    """
    dtype_out = []
    for dtype in dtypes:
        for name in dtype.names:
            dtype_out.append((name, dtype[name]))
    return np.dtype(dtype_out)
