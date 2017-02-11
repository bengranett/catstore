# -*- coding: utf-8 -*-
#!/usr/bin/env python

import logging
import numpy as np
import h5py
import fitsio
from sklearn.neighbors import KDTree
from pypelid.utils import sphere, misc

class Catalogue(object):
	""" Base catalogue class. Internal to Pypelid. """
	logger = logging.getLogger(__name__)

	_coordinate_system = 'equatorial'
	_required_columns = ()
	_required_meta = ()
	_immutable_attributes = ('_coordinate_system', '_required_columns', '_required_meta', '_immutable_attributes')

	def __init__(self, data=None, metadata=None, **attrs):
		"""
		Parameters
		----------
		data : recarray
			structured array, rows are objects, columns are object properties, which include at least:
			x, y, mag, ra, dec
		metadata : dict
			dictionary of metadata, should include at least:
			catalogue name, zone_resolution, zone_list, a centre tuple
		"""
		# process the input meta data
		meta = {}
		if metadata is not None:
			for key, value in metadata.items():
				meta[key] = value
		for key, value in attrs.items():
			if key not in ['imagecoord','skycoord']:
				meta[key] = value

		for key in self._required_meta:
			if key not in meta:
				raise Exception('Meta key %s not found. It is needed in Catalogue metadata!'%key)

		#if len(meta) > len(self._required_meta):
		#	logging.warning('Some of the metadata you are passing to Catalogue is not used.')

		self.__dict__['_meta'] = meta

		# Save column names for practicality
		self.columns = None

		# Load the data
		self.__dict__['_data'] = {}
		if data is not None:
			self.load(data)

		self.__dict__['_spatial_key'] = None
		try:
			self._data['imagecoord']
			self._spatial_key = 'imagecoord'
		except (KeyError, ValueError):
			try:
				self._data['skycoord']
				self._spatial_key = 'skycoord'
			except KeyError:
				if 'imagecoord' in attrs.keys():
					self._spatial_key = 'imagecoord'
				elif 'skycoord' in attrs.keys():
					self._spatial_key = 'skycoord'
				else:
					self.logger.error("Need imagecoord or skycoord to make spatial queries.")
				if data is None:
					self.load({self._spatial_key: attrs[self._spatial_key]})
				else:
					raise
		self.logger.debug("Using %s for spatial index", self._spatial_key)

	def __getattr__(self, key):
		""" Return columns by name 

		Columns may be accessed by catalogue.alpha

		Parameters
		----------
		key

		"""
		# if the attribute is a class variable, return it
		try:
			return self.__dict__[key]
		except KeyError:
			pass

		# if the attribute is a column name in the data table, return the column
		try:
			return self.__dict__['_data'][key]
		except KeyError as e:
			raise AttributeError(e.message)
		except IndexError as e:
			raise AttributeError(e.message)
		except ValueError as e:
			raise AttributeError(e.message)

	def __getitem__(self, key):
		return self.__getattr__(key)

	def __setattr__(self, key, value):
		""" Set a catalogue attribute.  

			Raises CatalogueError exception if the variable name exists in the data table. 

		Parameters
		----------
		key
		value

		"""
		try:
			self.__dict__['_data'][key]
			raise CatalogueError("Data table is read-only.")
		except KeyError:
			pass
		except ValueError:
			pass
		if key in self._immutable_attributes:
			raise CatalogueError(str(key) + ' is read-only.')

		# otherwise set the class variable
		self.__dict__[key] = value

	def __str__(self):
		""" helpful string representation """
		description = "Catalogue of " + str(len(self._data)) + " objects."
		return description

	def __len__(self):
		""" Return length of catalogue """
		return len(self._data)

	def __contains__(self, key):
		""" Check if we have the given field key """
		if self._data.dtype.fields is None: return False
		return self._data.dtype.fields.has_key(key)

	def _convert_CatStore(self, store):
		""" """
		columns = [name for name in store._h5file.get_columns()]

		data = {}
		for cat in store:
			for name in columns:
				if not data.has_key(name):
					data[name] = []
				data[name].append(getattr(cat, name))

		for name in columns:
			data[name] = np.concatenate(data[name])

		self.__dict__['_data'] = data

	def load(self, data):
		""" Import the data array into the Catalogue class.

			Parameters
			----------
			data : dict or numpy.recarray
				the input data table
		"""

		# Test if conversion to structured array is needed

		convert = False

		try:
			self._convert_CatStore(data)
			return
		except:
			pass

		try:
			# Check if the input data is a structured array
			if not data.dtype.fields:
				raise TypeError('The Catalogue class must be loaded with a dictionary-like structure or numpy structured array!')
			else:
				# Save column names for practicality
				self.columns = list(data.dtype.names)
		except AttributeError:

			# Check if the input data is a dict
			try:
				data.items()
			except AttributeError:
				raise TypeError('The Catalogue class must be loaded with a dictionary-like structure or numpy structured array!')
			else:
				convert = True
				# Save column names for practicality
				self.columns = list(data.keys())

		# Make sure the needed columns are loaded
		for column in self._required_columns:
			try:
				data[column]
			except KeyError:
				raise Exception('Data array is missing a required column: %s'%column)

		# Finally assign the dictionary to the class instance attribute
		if convert:
			self.__dict__['_data'] = misc.dict_to_structured_array(data)
		else:
			self.__dict__['_data'] = data

	def dump(self, filename):
		""" Dump the data array to a Hickle file. """
		import hickle
		hickle.dump(filename, self.__dict__['_data'])

	def get_data(self):
		""" Return the data array."""
		return self.__dict__['_data']

	def build_tree(self):
		""" Initialize the data structure for fast spatial lookups.

		"""
		xy = self.__dict__['_data'][self._spatial_key]
		self._lookup_tree = KDTree(xy)

	def query_disk(self, x, y, radius=1.):
		""" Find neighbors to a given point (ra, dec).

		Parameters
		----------
		x 
			center x  
		y
			center y
		radius : optional
			radius (default=1.)

		Returns
		-------
		?
			indices of objects in selection

		"""
		try:
			self._lookup_tree
		except AttributeError:
			self.build_tree()

		if self._lookup_tree is None: self.build_tree()

		matches = self._lookup_tree.query_radius(np.transpose([x,y]), radius)
		return matches

	def query_box(self,  cx, cy, width=1, height=1, pad_x=0.0, pad_y=0.0, orientation=0.):
		""" Find objects in a rectangle.

		Parameters
		----------
		x : array
			center x
		y : array
			center y
		width : float or numpy.array
			width
		height : float or numpy.array
			height
		pad_x : float
			add this padding to width
		pad_y : float
			add this padding to height
		orientation : float
			position angle of the box

		Returns
		-------
		results : list of lists
			list of indices of objects in selection

		"""
		try:
			len(cx)
		except TypeError:
			cx = np.array([cx])
			cy = np.array([cy])
		theta = misc.torad(orientation)
		costheta = np.cos(theta)
		sintheta = np.sin(theta)

		r = np.sqrt(width**2 + height**2)/2.

		matches = self.query_disk(cx,cy,r)

		data_x, data_y = np.transpose(self.__dict__['_data'][self._spatial_key])

		# Allow width and height to be arrays
		try:
			nh = len(height)
		except TypeError:
			h = height
			nh = 1
		try:
			nw = len(width)
		except TypeError:
			w = width
			nw = 1

		# Loop through coords and find matches ??
		results = []
		for i, match in enumerate(matches):
			if nh>1: h = height[match]
			if nw>1: w = width[match]

			dx = data_x[match] - cx[i]
			dy = data_y[match] - cy[i]
			dxt = dx * costheta - dy * sintheta
			dyt = dx * sintheta + dy * costheta

			sel_x = np.abs(dxt) < (w/2. + pad_x)
			sel_y = np.abs(dyt) < (h/2. + pad_y)
			sel = np.where(sel_x & sel_y)

			results.append(np.take(match, sel[0]))

		if len(results)==1:
			return results[0]
		return results


	def plot(self, nplot=10000, **plotparams):
		""" Make a cartesian image coordinates scatter plot using matplotlib.

		Parameters
		----------
		nplot : int
			Maximum number of objects to show in the plot.

		"""
		xy = self.__dict__['_data'][self._spatial_key]

		# Subsample a big catalogue
		if len(self) > nplot:
			index = np.arange(len(self))
			index = np.random.choice(index, nplot)
			# Coordinates in radians
			x,y = np.transpose(xy[index])
		else:
			x,y = np.transpose(xy)

		import pylab
		pylab.scatter(x,y,**plotparams)

class CatalogueError(Exception):
	pass
	
if __name__ == '__main__':

	# Test loading with a structured array
	data = np.array(np.random.randn(10,3), dtype=[('x', float), ('y', int), ('z', float)])
	cat = Catalogue()
	cat._cat_meta = 'hello'
	cat.load(data)
	print "recarray columns: " + str(cat.columns)
	print cat.__dict__['_data'].dtype

	# Test loading with a dict
	data = {'x': np.random.randn(10), 'y': np.random.randn(10), 'z': np.random.randn(10)}
	cat = Catalogue()
	cat._cat_meta = 'hello'
	cat.load(data)
	print "dict columns: " + str(cat.columns)
	print cat.__dict__['_data'].dtype

	# Test loading with an HDF5 Group of datasets
	#cat = Catalogue()
	#cat._cat_meta = 'hello'
	#data = None
	#cat.load(data)
	#print "recarray columns: " + str(cat.columns)
	#print cat.__dict__['_data'].dtype
