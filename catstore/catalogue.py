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
			meta[key] = value

		for key in self._required_meta:
			if key not in meta:
				raise Exception('Meta key %s not found. It is needed in Catalogue metadata!'%key)

		if len(meta) > len(self._required_meta):
			logging.warning('Some of the metadata you are passing to Catalogue is not used.')

		self.__dict__['_meta'] = meta

		# Load the data
		self.__dict__['_data'] = {}
		if data is not None:
			self.load(data)

		self.__dict__['_spatial_key'] = None
		print self._spatial_key
		try:
			self._data['imagecoord']
			self._spatial_key = 'imagecoord'
		except KeyError:
			try:
				self._data['skycoord']
				self._spatial_key = 'skycoord'
			except KeyError:
				self.logger.warning("Need imagecoord or skycoord to make spatial queries.")
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

	def load(self, data):
		""" Import the data array """
		try:
			if not data.dtype.fields:
				raise TypeError('The Catalogue class must be loaded with a structured array!')
		except AttributeError:
			try:
				data.items()
			except AttributeError:
				raise TypeError('The Catalogue class must be loaded with a dictionary-like structure or numpy structured array!')

		for column in self._required_columns:
			try:
				data[column]
			except KeyError:
				raise Exception('Data array is missing a required column: %s'%column)

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
			self.lookup_tree
		except AttributeError:
			self.build_tree()

		if self.lookup_tree is None: self.build_tree()

		matches = self.lookup_tree.query_radius(np.transpose([x,y]), radius)
		return matches

	def query_box(self,  cx, cy, width=1, height=1, pad_x=0.0, pad_y=0.0, orientation=0.):
		""" Find objects in a rectangle.

		Parameters
		----------
		x : array
			center x
		y : array
			center y
		width : float
			width
		height : float
			height
		pad_x : float
			add this padding to width
		pad_y : float
			add this padding to height
		orientation : float
			position angle of the box

		Returns
		-------
		? : list
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

		data_x, data_y = np.transpose(self.__dict__['_data'][_spatial_key])

		results = []
		for i, match in enumerate(matches):
			dx = data_x[match] - cx[i]
			dy = data_y[match] - cy[i]
			dxt = dx * costheta - dy * sintheta
			dyt = dx * sintheta + dy * costheta

			sel_x = np.abs(dxt) < (width/2. + pad_x)
			sel_y = np.abs(dyt) < (height/2. + pad_y)
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
		xy = self.__dict__['_data'][_spatial_key]

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
	cat = Catalogue()
	cat._cat_meta = 'hello'
	print cat._immutable_attributes

