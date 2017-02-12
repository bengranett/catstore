import sys
import numpy as np
import os
import tempfile
import pypelid.utils.hdf5tools as hdf5tools
import logging

logging.basicConfig(level=logging.DEBUG)


def check_hashing(algo='blake2', hash_length=32, hash_info_len=2):
    filename = tempfile.NamedTemporaryFile(delete=False).name+".pypelid"

    storage = hdf5tools.HDF5Catalogue(filename, 'w',
                preallocate_file=False,
                hash_algorithm=algo,
                hash_length=hash_length,
                hash_info_len=hash_info_len)

    storage.close()
    logging.debug("wrote %s", filename)
    hdf5tools.validate_hdf5_file(filename, hash_info_len=hash_info_len, hash_algo_len=1)
    os.unlink(filename)


def test_hashing():
    yield check_hashing, 'blake2', 2
    yield check_hashing, 'blake2', 32
    yield check_hashing, 'blake2', 64
    yield check_hashing, 'blake2', 128

    yield check_hashing, 'md5', 2
    yield check_hashing, 'md5', 32
    yield check_hashing, 'md5', 64
    yield check_hashing, 'md5', 128
    yield check_hashing, 'md5', 255
