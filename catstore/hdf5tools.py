""" hdf5tools.py """
import signal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import socket
import numpy as np
from pypelid.utils import misc
import h5py
import StringIO
import logging
import textwrap
import copy
import time
import string

import catstore
from pyblake2 import blake2b

Group = h5py._hl.group.Group

hash_algorithms = {
				'M': 'md5',
				'B': 'blake2'
				}

def get_hasher(algorithm):
	try:
		algorithm_name = hash_algorithms[algorithm]
	except KeyError:
		algorithm_name = algorithm

	if algorithm_name == 'blake2':
		from pyblake2 import blake2b
		hasher = blake2b(digest_size=digest_size)
	elif algorithm_name == 'md5':
		from hashlib import md5
		hasher = md5()
	else:
		raise HDF5CatError("Unrecognized hash algorithm (%s)" % algorithm)
	return hasher

def hash_it(filename, hash_length=32, reserved=8, skip=None,
			store_hash_in_file=False, chunk_size=1048576,
			algorithm='blake2', max_file_size=1024):
	""" Compute hash of a file.

	If the hash is stored in the first few bytes of the file these bytes can be
	skipped by setting store_hash_in_file.

	Inputs
	------
	filename : str
		name of file to operate on
	hash_length : int
		number ascii characters (1 byte each) reserved for the hash at the start of the file.
	reserved :  int
		skip a number of bytes at the start of the file (only used if store_hash_in_file is True)
	store_hash_in_file : bool
		the hash will be inserted at the start of the file, so start hashing after hash_length.
	chunk_size : int
		hash chunk size
	algorithm :  str
		hashing algorithm can be 'blake2' or 'md5'
	max_file_size : float
		hashing will stop after processing up to max_file_size in MB.

	Outputs
	-------
	str : digest
	"""
	digest_size = hash_length // 2   # Two hex characters per byte
	assert digest_size > 0

	try:
		algorithm_name = hash_algorithms[algorithm]
	except KeyError:
		algorithm_name = algorithm

	if algorithm_name == 'blake2':
		from pyblake2 import blake2b
		hasher = blake2b(digest_size=digest_size)
	elif algorithm_name == 'md5':
		from hashlib import md5
		hasher = md5()
	else:
		raise HDF5CatError("Unrecognized hash algorithm (%s)" % algorithm)

	#logging.debug("Hash algorithm: %s %i", algorithm, hasher.digest_size)

	if hasher.digest_size > digest_size:
		logging.warning("Digest length for algorithm %s is %i hex characters.  Digest will be truncated to %i",
							algorithm, hasher.digest_size * 2, hash_length)

	t0 = time.time()

	#logging.debug("reserved %i",reserved)

	num_chunks = (max_file_size * 1024**2) // chunk_size

	with open(filename, 'rb') as f:
		if store_hash_in_file:
			f.seek(hash_length + reserved)

		count = 0
		#logging.debug("reading file")
		while count < num_chunks:
			chunk = f.read(chunk_size)
			if chunk == '':
				break
			hasher.update(chunk)
			count += 1
		#logging.debug("done reading file")

	size = count * chunk_size / 1024.**2
	#logging.debug("Hashed %3.1f MB in %f sec", size, time.time() - t0)

	digest = hasher.hexdigest()[:hash_length]
	return digest


def read_hdf5_hash(filename, skip=7, hash_info_len=2, hash_algo_len=1):
	""" Read the hash string from the beginning of the file.

	The first bytes in the file specified by skip are reserved.
	The byte at skip+1 stores the hash_length

	Parameters
	----------
	filename : str
		path to file
	skip : int
		number of bytes reserved before the hash data in the file.
	hash_info_len : int
		Number of bytes used to store the hash length (usually 1).

	Note
	----
	The total number of bytes before the hash is skip+hash_info_len
	"""
	with open(filename, 'rb') as f:
		f.seek(skip)
		hash_length = int(f.read(hash_info_len), base=16)
		hash_algo = f.read(hash_algo_len)
		digest = f.read(hash_length)
	#logging.debug("hash length %i, hash_algo %s, digest %i", hash_length, hash_algo, len(digest))

	# remove null characters in case the digest is shorter than the reserved space.
	digest = string.translate(digest, None, chr(0))

	return digest, hash_algo, hash_length


def validate_hdf5_hash(filename, skip=7, hash_info_len=2, hash_algo_len=1):
	""" Check if the hash matches the file.

	Parameters
	----------
	filename : str
		path to file
	skip : int
		number of bytes reserved before the hash data in the file.
	hash_info_len : int
		Number of bytes used to store the hash length (usually 1).

	Returns
	------
	Return True if hash matches

	Raises
	------
	FileValidationError if hash does not match.

	Note
	----
	The total number of bytes before the hash is skip+hash_info_len
	"""
	digest_read, hash_algo_code, hash_length = read_hdf5_hash(filename, skip=skip,
											hash_info_len=hash_info_len, hash_algo_len=hash_algo_len)

	digest_comp = hash_it(filename, hash_length, reserved=skip+hash_info_len+hash_algo_len,
							store_hash_in_file=True, algorithm=hash_algo_code)

	if digest_read == digest_comp:
		return True

	raise HDF5CatError("Error reading file %s: hash validation failed.  Computed %s but expected %s."%(filename, digest_comp, digest_read))


def validate_hdf5_stamp(filename, expected='pypelid'):
	""" Read the first few bytes of the file.

	Parameters
	----------
	filename : str
		path to file
	expected : str
		prefice expected (usually 'pypelid').

	Returns
	------
	Return True if hash matches

	Raises
	------
	FileValidationError if string does not match.
	"""
	with open(filename, 'rb') as f:
		stamp = f.read(len(expected))

	if stamp == expected:
		return True

	raise HDF5CatError("Error reading file %s: stamp validation failed.  Read %s but expected %s."%(filename, stamp, expected))


def validate_hdf5_file(filename, check_hash=True, require_hash=True, official_stamp='pypelid',
					hash_info_len=2, hash_algo_len=1):
	""" Read the first few bytes of the file.

	Parameters
	----------
	filename : str
		path to file
	check_hash : bool
		check hash
	require_hash : bool
		require valid hash for validation
	official_stamp : str
		prefice expected (usually 'pypelid').

	Returns
	------
	Return True if hash matches, False otherwise

	Raises
	------
	FileValidationError is raised if hash does not match and require_hash is True
	"""
	# test that the pypelid stamp is in the header
	validate_hdf5_stamp(filename, official_stamp)

	# test that the hash matches
	if check_hash:
		try:
			validate_hdf5_hash(filename, hash_info_len=hash_info_len, hash_algo_len=hash_algo_len)
		except HDF5CatError:
			logging.warning("%s: hash validation failed.", filename)
			if require_hash:
				raise
			return False
	return True


class HDF5Catalogue(object):
	""" HDF5 catalogue backend.

	Parameters
	----------
	filename : str
		name of file to operate on
	mode : str
		Open file for reading or writing ('r', 'w')
	preallocate_file : bool
		data arrays will be initialized zero with size given by the max number
		of objects in the group.

	Optional parameters
	------------------
	chunk_size : int
		number of elements in chunk (should correspond to 1KB to 1MB)
	hash_length : int
		number of bytes reserved for the hash at the start of the file.
	header_bytes : int
		bytes reserved for the header at the start of the file.
	stamp : str
		string to print at the beginning of the header.
	hashit : bool
		compute the hash
	compression : dict
		dictionary of paramters to be passed to create_dataset (may be used to
		enable compression).
	"""
	logger = logging.getLogger(__name__)

	# These default parameters will be deep-copied in on initialization.
	_default_params = {
		'special_group_names': {'data': 'data', 'columns': 'columns',
								'units': 'units', 'description': 'description'},
		'header_line_width': 80,
		'chunk_size': None,
		'hash_length': 32,
		'header_bytes': 8192,
		'stamp': 'pypelid',
		'hashit': True,
		'hash_info_len': 2,
		'hash_algo_len': 1,
		'hash_algorithm': 'blake2',
		'preallocate_file': True,
		'compression': {},
		'title': 'PYPELID CATALOGUE',
		'overwrite': False,
		'libver': 'latest',
		'driver': None,
		'rdcc_nbytes': 1024**3,
		'rdcc_w0': 0.75,
		'rdcc_nslots': 52709,
	}


	attributes = {}

	def __init__(self, filename, mode='a', preallocate_file=True, **input_params):
		""" """
		self.filename = filename

		self.params = copy.deepcopy(self._default_params)

		self.params['preallocate_file'] = preallocate_file

		self.attributes = {}
		self.data = None
		self.dtypes = []

		for key, value in input_params.items():
			try:
				self.params[key] = value
			except KeyError:
				self.logger.warning("Unrecognized argument passed to HDF5Catalogue (%s)" % key)

		self._check_inputs()

		self.readonly = False
		if not mode in ('r', 'w', 'a'):
			raise HDF5CatError("Invalid argument: mode must be one of ('r', 'w', 'a')")

		if mode == 'w':
			if os.path.exists(filename) and not self.params['overwrite']:
				raise HDF5CatError("File %s exists.  Will not overwrite."%filename)

		logging.debug("HDF5 chunk_size: %s", self.params['chunk_size'])

		h5params = {
			'libver': self.params['libver'],
			'driver': self.params['driver'],
			'rdcc_nbytes': self.params['rdcc_nbytes'],
			'rdcc_w0': self.params['rdcc_w0'],
			'rdcc_nslots': self.params['rdcc_nslots']
		}
		for key,value in h5params.items():
			logging.debug("HDF5 %s: %s", key, value)

		if mode == 'r':
			self.readonly = True
			self.storage = h5py.File(filename, mode=mode, **h5params)
		else:
			self.storage = h5py.File(filename, mode=mode,
									userblock_size=self.params['header_bytes'], **h5params)

		self.attributes = self.storage.attrs

		self.data = self.storage.require_group(self.params['special_group_names']['data'])
		self.metadata = self.data.attrs

		if mode is not 'r':
			# Store metadata about file allocation
			if 'preallocate_file' not in self.metadata:
				self.metadata['preallocate_file'] = self.params['preallocate_file']

			# Flag to indicate completion of file allocation
			if 'allocation_done' not in self.metadata:
				self.metadata['allocation_done'] = False

			# Flag to indicate that all data that was preallocated has been loaded
			if 'done' not in self.metadata:
				self.metadata['done'] = False


	def _check_inputs(self):
		""" Sanity check arguments. """
		assert self.params['hash_length'] > 1 and self.params['hash_length'] < 256
		assert self.params['hash_info_len'] > 0
		assert self.params['hash_algo_len'] == 1
		assert self.params['hash_algorithm'] in hash_algorithms.values()
		assert self.params['header_line_width'] > 0
		assert self.params['header_bytes'] > 0
		assert self.params['header_bytes'] > 0
		assert self.params['preallocate_file'] in (True, False)
		assert self.params['hashit'] in (True, False)
		assert self.params['chunk_size'] is None or self.params['chunk_size'] > 0
		assert isinstance(self.params['stamp'], str)
		assert isinstance(self.params['title'], str)
		assert self.params['libver'] in (None,'latest','earliest')
		assert self.params['driver'] in (None,'core','family')

	def __enter__(self):
		""" """
		return self

	def __exit__(self, type, value, traceback):
		""" ensure the hdf5 file is closed. """
		# Ignore INT and TERM signals while closing the file
		state = {}
		for sig in (signal.SIGINT, signal.SIGTERM):
			state[sig] = signal.signal(sig, signal.SIG_IGN)

		self.logger.debug("closing file `%s`...", self.filename)
		self.close()
		self.logger.debug("file closed `%s`.", self.filename)

		# reset signal handler
		for sig, handler in state.items():
			signal.signal(sig, handler)

	def __str__(self):
		""" """
		return "< {}: {} >".format(self.__class__.__name__, self.filename)

	def __getattr__(self, key):
		""" """
		return self.storage.attrs[key]

	def __getitem__(self, key):
		""" """
		return self.storage[key]

	def get_data(self):
		""" """
		self.logger.warning("HDF5Catalogue.get_data() is depricated. Use .data attribute instead.")
		try:
			return self.storage[self.params['special_group_names']['data']]
		except KeyError:
			raise HDF5CatError("File has no data")

	def get_data_group(self, group_name):
		""" """
		return self.storage[self.params['special_group_names']['data']][str(group_name)]

	def update_attributes(self, attributes=None, **attrs):
		"""Add attributes to the HDF5 file.
		attributes - dictionary of attributes 
					 (may also be specified as named arguments)
		"""
		if self.readonly:
			raise HDF5CatError("File loaded in read-only mode.")
		if attributes is not None:
			for key, value in attributes.items():
				self.storage.attrs[key] = value
				self.logger.debug("HDF5 set %s %s %s", key, value, self.storage.attrs[key])
		for key, value in attrs.items():
			self.storage.attrs[key] = value

	def preallocate_groups(self, group_names, nmax, dtypes):
		""" Create groups containing preallocated datasets.

		Parameters
		----------
		group_names : int
			Names of groups to create (may be a single name or a list of names)
		nmax : int
			Number of data rows in the group (may be a single int or a list corresponding to group_names)
		dtypes : list
			List of tuples that are recognized by np.dtype
		"""
		if isinstance(dtypes, np.dtype):
			dtype_list = []
			for name in dtypes.names:
				dtype_list.append(np.dtype([(name, dtypes[name])]))
			dtypes = dtype_list

		if not misc.check_is_iterable(group_names):
			group_names = [group_names]
		if not misc.check_is_iterable(nmax):
			nmax = [nmax]

		for i, group_name in enumerate(group_names):
			group_name = str(group_name)
			try:
				self.get_data_group(group_name)
				raise HDF5CatError("Attempted to pre-allocate a group that already exists.")
			except KeyError:
				pass

			# Expand the full path to the group
			group_path = self._get_group_path(group_name)

			# Create the group
			group = self.storage.require_group(group_path)

			for t in dtypes:
				try:
					dtype = np.dtype(t)
				except TypeError:
					dtype = np.dtype([t])

				name = dtype.names[0]

				# create an empty dataset
				self.create_dataset(group, name, dtype=dtype[0], nmax=nmax[i])

			group.attrs['count'] = 0
			group.attrs['nmax'] = nmax[i]

		self.metadata['allocation_done'] = True
		self.dtypes = dtypes

	def create_dataset(self, group, name, dtype=None, arr=None, nmax=None):
		""" Create a new dataset within a data group.

		If arr is None, the dataset will be preallocated with nmax elements.

		Parameters
		----------
		group : str
			group name
		name : str
			dataset name
		dtype : numpy dtype
			dtype for the dataset (not used if arr is given)
		arr : numpy array
			data array
		nmax : int
			max number of elements for preallocation

		Raises
		------
		"""
		if nmax is None and 'nmax' in group.attrs:
			nmax = group.attrs['nmax']

		if nmax is not None and 'nmax' in group.attrs:
			if group.attrs['nmax'] != nmax:
				raise HDF5CatError("Cannot update group %s because supplied nmax is not consistent with group attributes."%group)

		if arr is None:
			arr = np.zeros(nmax, dtype=dtype)

		if self.params['chunk_size'] is None:
			chunkshape = None
		else:
			chunkshape = list(arr.shape)
			chunkshape[0] = self.params['chunk_size']
			chunkshape = tuple(chunkshape)
		maxshape = list(arr.shape)
		maxshape[0] = nmax
		maxshape = tuple(maxshape)

		if chunkshape and nmax < chunkshape[0]:
			# self.logger.debug("Dataset is too small for chunking (chunkshape %s, nmax %s)", chunkshape, nmax)
			chunkshape = None

		if not chunkshape:
			maxshape = None

		# self.logger.debug("create dataset %s, shape %s, max %s, chunk %s", name, arr.shape, maxshape, chunkshape)

		group.create_dataset(name, data=arr, maxshape=maxshape, chunks=chunkshape, **self.params['compression'])
		# self.logger.debug("dataset %s, chunks %s",name, group[name].chunks)
		# determine the number of elements per data row if 2 dimensional
		if len(arr.shape) <= 1:
			dim = len(arr.shape)
		else:
			dim = np.prod(arr.shape[1:])

		try:
			dtype_string = str(arr.dtype[0])
		except KeyError:
			dtype_string = str(arr.dtype)
		self.update_metagroup(self.params['special_group_names']['columns'], {name: "%i %s" % (dim, dtype_string)})

	def append_dataset(self, group, name, arr):
		""" Append a data array to a dataset. 

		Parameters
		----------
		group : str
			group name
		name : str
			dataset name
		arr : numpy array
			data array
		"""
		dim = group.attrs['count']

		if name not in group:
			return self.create_dataset(group, name, arr=arr)

		if not self.params['preallocate_file']:
			# if the group already exists, but the rows are not specified append the array to the dataset.
			group[name].resize(dim + arr.shape[0], axis=0)
		if dim + len(arr) > group[name].shape[0]:
			raise HDF5CatError("Cannot update dataset %s.%s Allocated dataset is too small to fit the input array."%(group_name,name))
		group[name][dim:dim + len(arr)] = arr
		# self.logger.debug("appending to dataset: %s %s chunky:%s",name,group[name].shape,group[name].chunks)

	def append_group(self, group_name, data, index=None):
		""" Append data to a group. 

		Parameters
		----------
		group : group object
			group
		data : numpy struc array or dictionary like object
			data array
		"""
		# Expand the full path to the group
		group_path = self._get_group_path(str(group_name))

		# Create the group
		group = self.storage.require_group(group_path)

		if 'count' not in group.attrs:
			group.attrs['count'] = 0

		if 'done' not in group.attrs:
			group.attrs['done'] = False

		try:
			columns = data.keys()
		except AttributeError:
			columns = data.dtype.names

		count = None

		for name in columns:
			if index is not None:
				sub = data[name][index]
			else:
				sub = data[name]

			if count is None:
				count = len(sub)
			else:
				assert count == len(sub)

			self.append_dataset(group, name, sub)

		group.attrs['count'] += count

		# if the expected data has been loaded, flag as done
		if self.metadata['preallocate_file']:
			if group.attrs['count'] == group.attrs['nmax']:
				group.attrs['done'] = True

		return count

	def append(self, group_arr, data):
		""" Append data across multiple groups.

		Parameters
		----------
		group_arr : numpy.ndarray
			a column corresponding in length to the number of rows in data
			giving a rule how to distribute the objects into the HDF5 groups
		data : dict or numpy structured array
			the data columns given as dictionary entries with columns names as keys
		"""
		# Check that we are allowed to write
		if self.readonly:
			raise HDF5CatError("File loaded in read-only mode.")

		# Loop over all the unique zone identifiers in the group list
		# optimisation comment: This is N=1 scan of the data column
		count = 0
		for zone in np.unique(group_arr):

			# Identify all the corresponding row indices
			# optimisation comment: This adds len(np.unique(group_list)) scans to N
			zone_index, = np.where(group_arr == zone)
			zone_size = len(zone_index)

			if zone_size == 0:
				continue

			count += self.append_group(zone, data, index=zone_index)

		self.check_data()

	def update_dataset(self, group, name, data, index, operation):
		""" Update indexed elements of a dataset.

		Parameters
		----------
		group : str
			group name
		name : str
			dataset name
		arr : numpy array
			data array
		index : numpy bool array
			indexes
		"""
		sel = np.zeros(len(group[name]), dtype=bool)
		sel[index] = True
		if operation == 'replace':
			group[name][sel] = data
		elif operation == 'sum':
			group[name][sel] = group[name][sel] + data
		else:
			raise Exception("unknown operation %s"%operation)

		# self.logger.debug("update %s dataset at specified indices: %s %s chunky:%s", operation, name,group[name].shape,group[name].chunks)

	def update_group(self, group_name, data, index, operation):
		""" update data in a group.

		Parameters
		----------
		group : group object
			group
		data : numpy struc array or dictionary like object
			data array
		"""
		# Expand the full path to the group
		group_path = self._get_group_path(str(group_name))

		# Create the group
		group = self.storage.require_group(group_path)

		try:
			columns = data.keys()
		except AttributeError:
			columns = data.dtype.names

		for name in columns:
			self.update_dataset(group, name, data[name], index, operation)

	def update(self, group_arr, data, index, operation='replace'):
		""" Update data across multiple groups.

		Parameters
		----------
		group_arr : numpy.ndarray
			a column corresponding in length to the number of rows in data
			giving a rule how to distribute the objects into the HDF5 groups
		data : dict or numpy structured array
			the data columns given as dictionary entries with columns names as keys
		"""
		# Check that we are allowed to write
		if self.readonly:
			raise HDF5CatError("File loaded in read-only mode.")

		# Loop over all the unique zone identifiers in the group list
		# optimisation comment: This is N=1 scan of the data column
		for zone in np.unique(group_arr):

			# Identify all the corresponding row indices
			# optimisation comment: This adds len(np.unique(group_list)) scans to N
			zone_index, = np.where(group_arr == zone)
			zone_size = len(zone_index)

			if zone_size == 0:
				continue

			ind = index[zone_index]

			self.update_group(zone, data[zone_index], ind, operation)


	def _get_group_path(self, group_name):
		""" Return the full path to the group """
		return '%s/%s'%(self.params['special_group_names']['data'], group_name)

	def check_data(self):
		""" Check if loaded data count matches preallocation.
		Set attribute flag if done.
		"""
		if not self.metadata['preallocate_file']:
			self.metadata['done'] = True
			return True

		done = True
		for group in self.data:
			try:
				done &= self.data[group].attrs['done']
			except KeyError:
				done = False
		self.metadata['done'] = done
		self.logger.debug("Data done flag: %s", self.metadata['done'])
		return done

	def write_data_hash(self, columns=None, validate=False, overwrite=False, digest_size=16):
		""" """
		fail = False
		data_hash = blake2b(digest_size=digest_size)
		for group in self.data:
			g = self.data[group]

			if columns is None:
				columns = g.keys()

			group_hash = blake2b(digest_size=digest_size)
			for dataset in columns:
				digest = blake2b(g[dataset][:],digest_size=digest_size).hexdigest()

				if validate:
					if g[dataset].attrs['_hash'] != digest:
						logging.warning("dataset has changed '%s/%s'", group, dataset)
						fail = True
					else:
						logging.debug(u"\u2714 '%s/%s'", group, dataset)
				else:
					if '_hash' in g[dataset].attrs and not overwrite:
						raise ValueError("Will not overwrite data hashes")
					g[dataset].attrs['_hash'] = digest
					logging.debug("wrote hash '%s/%s'", group, dataset)

				group_hash.update(digest)

			digest = group_hash.hexdigest()

			if validate:
				if g.attrs['_hash'] != digest:
					logging.warning("group has changed '%s'", group)
					fail = True
			else:
				g.attrs['_hash'] = digest
			data_hash.update(digest)

		digest = data_hash.hexdigest()

		if validate:
			if self.data.attrs['_hash'] != digest:
				logging.warning("data has changed")
				fail = True
		else:
			self.data.attrs['_hash'] = digest
			self.storage.attrs['_hash'] = digest

		return fail

	def validate_data(self, columns=None):
		""" """
		self.write_data_hash(columns=columns, validate=True)

	def update_metagroup(self, group_name, attributes, **attrs):
		""" Create a group to store metadata.

		group_name : str
			name of the meta data group
		attributes : dict
			metadata key-value pairs
		"""
		if self.readonly: raise HDF5CatError("File loaded in read-only mode.")
		group = self.storage.require_group(group_name)
		for key, value in attributes.items():
			group.attrs[key] = value
		for key, value in attrs.items():
			group.attrs[key] = value

	def get_metagroup(self, group_name):
		""" """
		return self.storage[group_name].attrs

	def update_units(self, attributes, **attrs):
		""" Update the units metadata group. """
		self.update_metagroup(self.params['special_group_names']['units'], attributes, **attrs)

	def update_description(self, attributes, **attrs):
		""" Update the description metadata group. """
		self.update_metagroup(self.params['special_group_names']['description'], attributes, **attrs)

	def get_units(self):
		""" Retrieve the units attributes """
		return self.get_metagroup(self.params['special_group_names']['units'])

	def get_description(self):
		""" Retrieve the description attributes """
		return self.get_metagroup(self.params['special_group_names']['description'])

	def get_columns(self):
		""" Retrieve the columns attributes """
		return self.get_metagroup(self.params['special_group_names']['columns'])

	def close(self):
		""" Write the HDF5 data to disk and then insert the human-readable header."""
		if self.readonly:
			self.storage.close()
			return

		self.storage.attrs['catstore_version'] = catstore.__version__
		self.storage.flush()
		self.storage.close()
		self.write_header()

	def format_attr(self, name, attrs):
		""" """
		header = StringIO.StringIO()
		header.write(self.horizline(name))

		for key, value in attrs.items():
			header.write("%16s: %s\n"%(key,value))

		header.write(self.horizline())

		return header.getvalue()

	def horizline(self, title=""):
		""" """
		start = "_%s" % title
		start += "_" * (self.params['header_line_width'] - len(start))
		start += "\n"
		return start

	def write_header(self):
		""" Insert a human-readable header at the start of the HDF5 file.
		"""
		if self.readonly:
			raise HDF5CatError("File loaded in read-only mode.")
		# open the file to read the attributes
		f = h5py.File(self.filename)

		reserved = len(self.params['stamp']) + self.params['hash_info_len'] + self.params['hash_algo_len']
		header_bytes = self.params['header_bytes'] - self.params['hash_length'] - reserved
		# self.logger.debug("HDF5 File %s has header block size %i.", self.filename, header_bytes)

		header = StringIO.StringIO()
		header.write(" " * header_bytes)
		header.seek(0)

		header.write("\n")
		header.write(("{:^%is}\n" % self.params['header_line_width']).format(self.params['title']))
		header.write(self.horizline())

		header.write(self.format_attr("file attributes", f.attrs))

		for group_name in f:
			if group_name in self.params['special_group_names'].values():
				continue
			if group_name.startswith("_"):
				continue
			header.write(self.format_attr(group_name, f[group_name].attrs))

		# write the column description lines
		header.write(self.horizline("data columns"))
		if self.params['special_group_names']['columns'] in f:
			for name, info in f[self.params['special_group_names']['columns']].attrs.items():
				try:
					unit = f[self.params['special_group_names']['units']].attrs[name]
				except:
					unit = ""
				try:
					desc = f[self.params['special_group_names']['description']].attrs[name]
				except:
					desc = ""
				message = "{:^16s}|{:^10s}|{:^10s}| ".format(name, unit, info)
				pad = min(self.params['header_line_width']//2, len(message))
				desc_lines = textwrap.wrap(desc,self.params['header_line_width']-pad)
				if len(desc_lines)>0:
					if len(message) + len(desc_lines[0]) > self.params['header_line_width']:
						message += "\n"+" "*pad  # go to next line.
					message += desc_lines[0] + "\n"
					for line in desc_lines[1:]:
						message += " "*pad + line + "\n"
				else:
					message += "\n"
				header.write(message)
		header.write(self.horizline())

		header.write(
			"This file was generated with CatStore v{version} by {user}@{hostname} at {date}.".format(
			version=catstore.__version__,
			user=os.getenv('USER'),
			hostname=socket.gethostname(),
			date=time.strftime('%H:%M:%S, %d %b %Y')
			)
		)

		# close the header
		head = header.getvalue()
		header.close()

		if len(head) > header_bytes:
			head = head[:header_bytes]
			self.logger.critical("HDF5 header truncated to %i characters!  Your header may be corrupted.", len(head))

		with open(self.filename, 'rb+') as f:
			f.seek(self.params['hash_length'] + reserved)
			f.write(head)

		if self.params['hashit']:
			digest = hash_it(self.filename, self.params['hash_length'],
								reserved=reserved, store_hash_in_file=True,
								algorithm=self.params['hash_algorithm'])

			# look up algorithm code
			algorithm_code = misc.dict_reverse_lookup(hash_algorithms, self.params['hash_algorithm'])

			# convert hash length to hex
			hash_len_code = format(self.params['hash_length'], str(self.params['hash_info_len']) + 'x')
			assert len(hash_len_code) == self.params['hash_info_len']
			# write the hash
			with open(self.filename, 'rb+') as f:
				f.write(self.params['stamp'])
				f.write(hash_len_code)
				f.write(algorithm_code)
				f.write(digest)

	def show(self, thing=None, pre="", level=0, datasets=False, attrs=False):
		""" Show what is in the file. """
		if thing is None:
			thing = self.storage
		try:
			thing.items()
		except AttributeError:
			return

		try:
			attrib = thing.attrs
		except:
			attrib = None
		if attrib is not None:
			if attrs:
				self.show(attrib, pre=pre, level=level, datasets=datasets, attrs=attrs)

		for key, value in thing.items():

			if level == 0:
				print("+ "),

			if isinstance(value, h5py.Group):
				s = str(value)
			elif isinstance(value, h5py.Dataset):
				if not datasets:
					continue
				if value.size == 1:
					s = value[()]
				else:
					s = str(value)
			else:
				s = str(value)

			print "%s%s%s: %s"%("   "*level, pre, key, s)
			self.show(value, pre+"\__",level+1, datasets=datasets, attrs=attrs)


class HDF5CatError(Exception):
	pass
