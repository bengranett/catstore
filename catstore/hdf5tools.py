""" hdf5tools.py """
import numpy as np
from pypelid.utils import misc
import h5py
import StringIO
import logging
import textwrap

def hash_it(filename, hash_length=32, reserved=8, skip=None, store_hash_in_file=False, chunk_size=1048576):
    """ Compute hash of a file.

    Inputs
    ------
    filename - name of file to operate on
    hash_length - number of bytes reserved for the hash at the start of the file.
    reserved - skip a number of bytes at the start of the file (only used if store_hash_in_file is True)
    store_hash_in_file - the hash will be inserted at the start of the file, so start hashing after hash_length.
    chunk_size - hash chunk size

    Outputs
    -------
    digest string
    """
    import time
    from pyblake2 import blake2b

    t0 = time.time()

    digest_size = hash_length//2   # Two hex characters per byte

    hasher = blake2b(digest_size=digest_size)

    with open(filename, 'rb') as f:
        if store_hash_in_file:
            f.seek(hash_length + reserved)

        count = 0
        logging.debug("reading file")
        while True:
            chunk = f.read(chunk_size)
            if chunk == '': break
            hasher.update(chunk)
            count += 1
        logging.debug("done reading file")

    size = count * chunk_size/1024.**2
    logging.debug("Hashed %3.1f MB in %f sec",size, time.time()-t0)

    return hasher.hexdigest()

def read_hdf5_hash(filename, skip=7, hash_info_len=1):
    """ Read the hash string from the beginning of the file. 
    The first bytes in the file specified by skip are reserved.
    The byte at skip+1 stores the hash_length
    """
    with open(filename, 'rb') as f:
        f.seek(skip)
        hash_length = ord(f.read(hash_info_len))
        digest = f.read(hash_length)

    return digest

def validate_hdf5_hash(filename, skip=7, hash_info_len=1):
    """ Check if the hash matches the file.

    Input
    -----
    filename 
    skip - number of bytes reserved before the hash data in the file.
           The hash digest will start at skip + 1.

    Output
    ------
    Return True if hash matches, False otherwise
    tuple (digest read, digest computed)

    """
    digest_read = read_hdf5_hash(filename, skip=skip, hash_info_len=hash_info_len)
    hash_length = len(digest_read)
    digest_comp = hash_it(filename, hash_length, reserved=skip+hash_info_len, store_hash_in_file=True)

    if digest_read == digest_comp:
        return True

    raise FileValidationError("Error reading file %s: hash validation failed.  Computed %s but expected %s."%(filename, digest_comp, digest_read))


def validate_hdf5_stamp(filename, expected='pypelid'):
    """ Read the first few bytes of the file. """
    with open(filename, 'rb') as f:
        stamp = f.read(len(expected))

    if stamp == expected:
        return True

    raise FileValidationError("Error reading file %s: stamp validation failed.  Read %s but expected %s."%(filename, stamp, expected))

def validate_hdf5_file(filename, check_hash=True, require_hash=True, official_stamp='pypelid'):
    """ """
    # test that the pypelid stamp is in the header
    validate_hdf5_stamp(filename, official_stamp)

    # test that the hash matches
    if check_hash:
        try:
            validate_hdf5_hash(filename)
        except FileValidationError:
            logging.warning("%s: hash validation failed.", filename)
            if require_hash:
                raise


class HDF5Catalogue(object):
    """ """
    # these are reserved group names

    DATA_GROUP = 'data'
    COLUMNS_GROUP = 'columns'
    UNITS_GROUP = 'units'
    DESCRIPTION_GROUP = 'description'

    _special_group_names = [DATA_GROUP, COLUMNS_GROUP, UNITS_GROUP, DESCRIPTION_GROUP]
    headerlinewidth = 80

    def __init__(self, filename, mode='a', chunk_size=1024, hash_length=32, header_bytes=4096, stamp='pypelid', 
                hashit=True, hash_info_len=1, preallocate_file=False, 
                compression={'compression':'lzf','shuffle':True,'fletcher32':True}):
        """     
        filename - name of file to operate on
        chunk_size - number of elements in chunk (should correspond to 1KB to 1MB)
        hash_length - number of bytes reserved for the hash at the start of the file.
        header_bytes - bytes reserved for the header at the start of the file.
        stamp - string to print at the beginning of the header.
        hashit - compute the hash
        preallocate_file - data arrays will be initialized zero with size given by the max number of objects in the group.
        compression - dictionary of paramters to be passed to create_dataset to enable compression.
        """
        self.filename = filename
        self.chunk_size = chunk_size
        self.hash_length = hash_length
        self.header_bytes = header_bytes
        self.stamp = stamp
        self.hashit = hashit
        self.hash_info_len = hash_info_len
        self.preallocate_file = preallocate_file
        self.compression = compression
        self.readonly = False
        if mode == 'r':
            self.readonly = True
            self.storage = h5py.File(filename, mode=mode)
        else:
            self.storage = h5py.File(filename, mode=mode, userblock_size=header_bytes)

        self.column_count = {}

    def __enter__(self):
        """ """
        return self

    def __exit__(self, type, value, traceback):
        """ ensure the hdf5 file is closed. """
        self.close()

    def get_attributes(self):
        """ """
        return self.storage.attrs

    def get_data(self):
        """ """
        return self.storage['data']

    def update_attributes(self, attributes=None, **attrs):
        """Add attributes to the HDF5 file.
        attributes - dictionary of attributes 
                     (may also be specified as named arguments)
        """
        if self.readonly: raise WriteError("File loaded in read-only mode.")
        if attributes is not None:
            for key, value in attributes.items():
                self.storage.attrs[key] = value
        for key, value in attrs.items():
            self.storage.attrs[key] = value

    def preallocate_groups(self, group_names, nmax, column_names, dtypes):
        """ Create groups containing preallocated datasets.

        Parameters
        ----------
        group_names : int
            Names of groups to create (may be a single name or a list of names)
        nmax : int
            Number of data rows in the group (may be a single int or a list corresponding to group_names)
        column_names : str list
            List of column names to be used when creating datasets.
        dtypes : type string of object
            List of dtype objects that are recognized by numpy with same length as column_names
        """
        # create an empty dataset with the given dtype
        data = {}
        for i, name in enumerate(column_names):
            data[name] = np.array([], dtype=dtypes[i])

        group_names = misc.ensurelist(group_names)
        nmax = misc.ensurelist(nmax)

        for i, group in enumerate(group_names):
            self.update_data(data, group, nmax[i])

    def update(self, group_arr, data):
        """ Update multiple groups.

        Parameters
        ----------
        group_arr : np.ndarray
            a column corresponding in length to the number of rows in data
            giving a rule how to distribute the objects into the HDF5 groups
        data : dict
            the data columns given as dictionary entries with columns names as keys

        Returns
        ------
        None
        """

        # Loop over all the unique zone identifiers in the group list
        # optimisation comment: This is N=1 scan of the data column
        for zone in np.unique(group_arr):

            # Identify all the corresponding row indices
            # optimisation comment: This adds len(np.unique(group_list)) scans to N
            index = np.where(group_arr == zone)

            # Call the update_data function to add to the group
            zond_data = {}
            for col in data.keys():
            	zone_data[col] = data[col][index]

            # Now update the group - this is uncertain if the groups don't exist
            self.update_data(zone_data, zone)

    def update_data(self, group_data, group_name=0, nmax=None):
        """ Add catalogue data belonging to a single group.
        group_data - dictionary
        group_name - default 0
        nmax - maximum number of objects in the group.  Only used if preallocate_file is enabled.
        """
        if self.readonly: raise WriteError("File loaded in read-only mode.")
        if self.preallocate_file and nmax is None:
            raise Exception("nmax must be specified to preallocate the hdf5 file.")
        group_name = '%s/%s'%(self.DATA_GROUP, group_name)

        group = self.storage.require_group(group_name)

        # count attribute will track the length of the data columns
        if not 'count' in group.attrs:
            group.attrs['count'] = 0

        column_info = {}

        length_check = None # this variable will be used to ensure that all data arrays have the same length.

        try:
            column_names = group_data.keys()
        except AttributeError:
            column_names = group_data.dtype.names

        for name in column_names:  # loop through column names
            arr = group_data[name]

            # Do a check of the length of the data array.
            if length_check is not None:
                if len(arr) != length_check:
                    raise Exception("The length of column %s does not match! length:%i expected:%i."%(name,len(arr),length_check))
            length_check = len(arr)

            if name in group:
                # if the group already exists append the array to the dataset.
                dim = group.attrs['count']
                if not self.preallocate_file: group[name].resize(dim+arr.shape[0], axis=0)
                if dim+len(arr) > group[name].shape[0]:
                    raise Exception("Allocated dataset is too small to fit the input array.")
                group[name][dim:dim+len(arr)] = arr
                #logging.debug("appending to dataset: %s %s chunky:%s",name,group[name].shape,group[name].chunks)
            else:
                # otherwise create a new dataset
                if self.chunk_size is None:
                    chunkshape = None
                else:
                    chunkshape = list(arr.shape)
                    chunkshape[0] = self.chunk_size
                    chunkshape = tuple(chunkshape)
                maxshape = list(arr.shape)
                maxshape[0] = nmax
                if self.preallocate_file:
                    group.create_dataset(name, data=np.zeros(maxshape,dtype=arr.dtype), maxshape=maxshape, chunks=chunkshape, **self.compression)
                    if len(arr) > group[name].shape[0]:
                        raise Exception("Allocated dataset is too small to fit the input array.")
                    group[name][:len(arr)] = arr
                else:
                    group.create_dataset(name, data=arr, maxshape=maxshape, chunks=chunkshape, **self.compression)
                #logging.debug("create dataset: %s %s %s chunky:%s",name,group[name].shape,maxshape,group[name].chunks)

            if not self.column_count.has_key(name):
                self.column_count[name] = 0
            self.column_count[name] += len(arr)

            dim = arr.size / len(arr)

            column_info[name] = "%i %s"%(dim, arr.dtype)

        # update the count attribute with the length of the data arrays
        group.attrs['count'] += len(arr)

        self.update_metagroup(self.COLUMNS_GROUP, column_info)

    def bulk_update_data(self, data):
        """Add catalogue data to the HDF5 file.
        data - dictionary structure: each key is a zone id and the vaue is the structured array holding the data.
        """
        for group_name, group_data in data.items():
            self.update_data(group_data, group_name=group_name)

    def update_metagroup(self, group_name, attributes, **attrs):
        """ Create a group to store metadata.
        group_name - name of the meta data group
        attributes - dictionary of meta data.
        """
        if self.readonly: raise WriteError("File loaded in read-only mode.")
        group = self.storage.require_group(group_name)
        for key,value in attributes.items():
            group.attrs[key] = value
        for key,value in attrs.items():
            group.attrs[key] = value


    def close(self):
        """ Write the HDF5 data to disk and then insert the human-readable header."""
        if self.readonly:
            self.storage.close()
            return

        gitenv = misc.GitEnv()
        self.storage.attrs['commit_hash'] = gitenv.hash
        self.storage.flush()
        self.storage.close()
        self.write_header()

    def format_attr(self, name, attrs):
        """ """
        header = StringIO.StringIO()
        header.write(self.horizline(name))

        for key,value in attrs.items():
            header.write("%16s: %s\n"%(key,value))

        header.write(self.horizline())

        return header.getvalue()

    def horizline(self, title=""):
        """ """
        start = "_%s"%title
        start += "_"*(self.headerlinewidth-len(start))
        start += "\n"
        return start

    def write_header(self):
        """ Insert a human-readable header at the start of the HDF5 file.
        """
        if self.readonly: raise WriteError("File loaded in read-only mode.")
        # open the file to read the attributes
        f = h5py.File(self.filename)

        reserved = len(self.stamp) + self.hash_info_len
        header_bytes = self.header_bytes - self.hash_length - reserved
        logging.debug("HDF5 File %s has header block size %i.", self.filename, header_bytes)

        header = StringIO.StringIO()
        header.write(" "*header_bytes)
        header.seek(0)

        header.write("\n")
        header.write(("{:^%is}\n"%self.headerlinewidth).format("PYPELID CATALOGUE"))
        header.write(self.horizline())

        header.write(self.format_attr("file attributes", f.attrs))

        for group_name in f:
            if group_name in self._special_group_names: continue
            header.write(self.format_attr(group_name, f[group_name].attrs))

        # write the column description lines
        header.write(self.horizline("data columns"))
        if self.COLUMNS_GROUP in f:
            for name,info in f[self.COLUMNS_GROUP].attrs.items():
                try:
                    unit = f[self.UNITS_GROUP].attrs[name]
                except:
                    unit = ""
                try:
                    desc = f[self.DESCRIPTION_GROUP].attrs[name]
                except:
                    desc = ""
                firstline = "{:^16s}|{:^10s}|{:^10s}| ".format(name,unit,info)
                pad = len(firstline)
                desc_lines = textwrap.wrap(desc,self.headerlinewidth-pad)
                message = firstline + desc_lines[0] + "\n"
                for line in desc_lines[1:]:
                    message += " "*pad + line + "\n"
                header.write(message)
        header.write(self.horizline())

        # write the git environment
        gitenv = misc.GitEnv()
        header.write(gitenv)

        # close the header
        head = header.getvalue()
        header.close()

        if len(head) > header_bytes:
            head = head[:header_bytes]
            logging.critical("HDF5 header truncated to %i characters!  Your header may be corrupted.", len(head))

        with open(self.filename, 'rb+') as f:
            f.seek(self.hash_length + reserved)
            f.write(head)

        if self.hashit and self.hash_length > 0 and self.hash_length < 256:
            digest = hash_it(self.filename, self.hash_length, reserved=reserved, store_hash_in_file=True)
            # write the hash
            with open(self.filename, 'rb+') as f:
                f.write(self.stamp)
                f.write(chr(self.hash_length))
                f.write(digest)

    def show(self, thing=None, pre="",level=0):
        """ Show what is in the file. """
        if thing is None:
            thing = self.storage
        try:
            thing.items()
        except AttributeError:
            return

        try:
            attrs = thing.attrs
        except:
            attrs = None
        if attrs is not None:
            self.show(attrs, pre=pre, level=level)

        for key, value in thing.items():

            if level==0:
                print("+ "),

            if isinstance(value, h5py.Group):
                s = str(value)
            elif isinstance(value, h5py.Dataset):
                if value.size==1:
                    s = value[()]
                else:
                    s = str(value)
            else:
                s = str(value)

            print "%s%s%s: %s"%("   "*level, pre, key, s)
            self.show(value, pre+"\__",level+1)

