import numpy as np

import catalogue
import pypelid.vm.healpix_projection as HP
import pypelid.utils.hdf5tools as hdf5tools
import pypelid.utils.misc as misc
import pypelid.utils.sphere as sphere

import logging

class CatalogueStore(object):
	""" Storage backend for loading catalogues from disk.

	The catalogues support spatial queries using spherical coordinates. 
	
	Full-sky catalogues may be loaded in spatial chunks called zones only when needed so that the entire
	catalogue is not stored in memory.
	"""

	# Define sky partition mode constants
	FULLSKY = 0
	HEALPIX = 1
	_required_attributes = {'partition_scheme': ('HEALPIX',),
							'zone_resolution': range(12),
							'zone_order': (HP.RING,HP.NEST)
							}
	_required_columns = ('skycoord',)

	def __init__(self, filename=None, zone_resolution=2, zone_order=HP.RING,
					check_hash=True, require_hash=True, official_stamp='pypelid'):
		""" Initialize the catalogue backend. """
		self.filename = filename
		self._trees = {}

		if filename is not None:
			self.init_pypelid_file(check_hash=check_hash, require_hash=require_hash, official_stamp=official_stamp)
			return
			
		self.zone_resolution = zone_resolution
		self.zone_order = zone_order
		self._hp_projector = HP.HealpixProjector(resolution = self.zone_resolution, order=self.zone_order)

	def init_pypelid_file(self, check_hash=True, require_hash=True, official_stamp='pypelid'):
		""" Check the input pypelid file and initialize. """
		hdf5tools.validate_hdf5_file(self.filename, check_hash=check_hash, require_hash=require_hash, 
									official_stamp=official_stamp)

		with hdf5tools.HDF5Catalogue(self.filename, mode='r') as h5file:
			# import all attributes in the file.
			self.metadata = {}
			for key,value in h5file.get_attributes().items():
				self.metadata[key] = value

			# ensure that required attributes are there with acceptable values.
			for key, options in self._required_attributes.items():
				try:
					assert(self.metadata[key] in options)
				except KeyError:
					raise Exception("Cannot load %s: attribute is missing: %s"%(self.filename,key))
				except AssertionError:
					raise Exception("Cannot load %s: invalid attribute: %s:%s (expected %s)"%(self.filename,key,self.metadata[key],options))

			self.zone_resolution = self.metadata['zone_resolution']
			self.zone_order = self.metadata['zone_order']

			# ensure that required datasets are there
			for column_name in self._required_columns:
				try:
					if column_name not in h5file.get_columns():
						raise Exception("Cannot load %s: data column is missing: %s"%(self.filename, column_name))
				except KeyError:
						raise Exception("Cannot load %s: column description group is missing"%(self.filename))

		self._hp_projector = HP.HealpixProjector(resolution = self.zone_resolution, order=self.zone_order)
		self._datastore = None

	def __enter__(self):
		""" """
		self._open_pypelid(self.filename)
		return self

	def __exit__(self, type, value, traceback):
		""" """
		self._close_pypelid()

	def _open_pypelid(self, filename):
		""" Load a pypelid catalogue store file. """
		self.h5file = hdf5tools.HDF5Catalogue(filename, mode='r')
		# access the data group
		self._datastore = self.h5file.get_data()
		return self.h5file

	def _close_pypelid(self):
		""" """
		self.h5file.close()

	def write(self, filename):
		""" Construct and write a new catalogue store file in pypelid format. """
		pass

	def index(self):
		""" Generate the indices for the catalogue eg healpix zones. """
		pass

	def _retrieve_zone(self, zone):
		""" Retrieve the data within the given zones."""
		key = str(zone)
		if key not in self._datastore:
			raise ZoneDoesNotExist()
		return self._datastore[key]

	def _which_zones(self, lon, lat, radius):
		""" Determine which zones overlap the points (lon,lat) within the radius."""
		zones = self._hp_projector.query_disc(lon, lat, radius)
		logging.debug("querying %f,%f... zones found: %s",lon,lat,zones)
		return zones

	def get_zones(self):
		""" Return a list of zone identifiers. """
		return self._datastore.keys()

	def get_data(self):
		""" A generating function that returns the hdf5 groups."""
		for zone in self.get_zones():
			yield self._datastore[zone]

	def _query_cap(self, clon, clat, radius=1.):
		""" Find neighbors to a given point (clon, clat).

		Inputs
		------
		clon - center longitude degrees
		clat - center latitude degrees
		radius - degrees

		Outputs
		-------
		indices of objects in selection
		"""

		mu_thresh = np.cos(radius * np.pi/180)

		xyz = sphere.lonlat2xyz(clon, clat)

		if misc.is_number(xyz[0]):
			xyz = np.transpose(xyz).reshape(1,-1)
		else:
			xyz = np.transpose(xyz)

		matches = {}
		for zone_i in self._which_zones(clon, clat, radius):
			try:
				data = self._retrieve_zone(zone_i)
			except ZoneDoesNotExist:
				continue
				
			# access longitude and latitude...
			lon,lat = np.transpose(data['skycoord'][:])
			cat_xyz = sphere.lonlat2xyz(lon, lat)

			mu = np.dot(xyz, cat_xyz)
			cut = mu > mu_thresh
			cut = np.sum(cut, axis=0)          # OR operation along the first dimension
			matches[zone_i] = np.where(cut)

		return matches

	def _query_box(self,  clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0):
		""" Find objects in a rectangle.
		Inputs
		------
		clon - center longitude
		clat - center latitude
		width - width (degrees)
		height - height (degrees)
		padding - add this padding to width and height (degrees)
		orientation - (degrees about center)

		Outputs
		-------
		indices of objects in selection
		"""
		r = np.sqrt(width**2 + height**2)/2.
		match_dict = self._query_cap(clon,clat,r)

		selection_dict = {}
		for zone,matches in match_dict.items():
			data = self._retrieve_zone(zone)
			lon,lat = np.transpose(data['skycoord'][:][matches])
			dlon,dlat = sphere.rotate_lonlat(lon, lat, [(orientation, clon, clat)], inverse=True)

			sel_lon = np.abs(dlon) < (width/2. + pad_ra)
			sel_lat = np.abs(dlat) < (height/2. + pad_dec)
			sel = np.where(sel_lon & sel_lat)

			selection_dict[zone] = np.take(matches, sel[0])

		return selection_dict


	def retrieve(self, clon, clat, width=1, height=1, pad_ra=0.0, pad_dec=0.0, orientation=0, transform=None):
		""" Select objects in a rectangle and return a projected catalogue object.

		Parameters
		----------
		clon : float
			center ra
		clat : float
			center dec
		width : float
			rectangle width (degrees)
		height : float
			rectangle height (degrees)
		pad_ra : float
			add this padding to width (degrees)
		pad_dec : float
			add this padding to dec (degrees)
		orientation : float
			rotation angle (degrees about center from North increasing toward the East)
		transform : function
			a function that will be called to map sky coordinates to cartesian coordinates.

		Returns
		-------
		ndarray : catalogue stored as a numpy structured array
		"""
		data_dict = {}

		matches = self._query_box(clon, clat, width, height, pad_ra, pad_dec, orientation)

		for zone, selection in matches.items():
			data = self._retrieve_zone(zone)
			for name,arr in data.items():
				if not name in data_dict:
					data_dict[name] = []
				data_dict[name].append(arr[:][selection])

		for name in data_dict.keys():
			data_dict[name] = np.concatenate(data_dict[name])

		if transform is not None:
			lon,lat = np.transpose(data_dict['skycoord'])
			ximage,yimage = transform(lon, lat)
			data_dict['imagecoord'] = np.transpose([ximage,yimage])

		# construct a structured array
		structured_arr = misc.dict_to_structured_array(data_dict)

		return structured_arr


	def plot(self, plot_every=10, s=5., lw=0., cmap='jet'):
		""" Create a Mollweide projected plot of the objects.

		Uses healpy for the spherical projection and matplotlib for the colormap.

		Parameters
		----------
		plot_every : int
				Downsample the points before plotting.
		s : float
				Marker size
		lw : float
				Line width
		cmap : colormap
				A matplotlib colormap.
		"""
		from matplotlib.cm import ScalarMappable
		from matplotlib.colors import Normalize
		import healpy as hp

		sc = ScalarMappable(Normalize(0,len(self.get_zones())), cmap=cmap)

		# initialize a mollweide map
		hp.mollview(np.zeros(12)+float('nan'), cbar=False)
		for data in self.get_data():
			lon,lat = np.transpose(data['skycoord'][:])

			# select a subset of points
			if plot_every>1:
				sel = np.random.choice(len(lon),len(lon)//plot_every)
				lon = lon[sel]
				lat = lat[sel]

			# color points based upon zone index
			c=sc.to_rgba(int(data.name.split("/")[-1]))

			# plot
			hp.visufunc.projscatter(lon, lat, c=c, s=s, lw=lw, lonlat=True)
		hp.graticule()


class ZoneDoesNotExist(Exception):
	pass