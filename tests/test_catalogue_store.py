import sys
import os
import tempfile
import numpy as np
import pypelid.utils.sphere as sphere
import logging

from catstore import catalogue_store

logging.basicConfig(level=logging.INFO)


def check_catalogue_store(n=100, zone_resolution=0):
	""" Quick demo of catalogue store.

		Generate mock data and write a catalogue store file.  
		Update the catalogue store with some new values.
		Open the file and check	that contents are correct.
	"""
	# temporary file to use for test
	filename = tempfile.NamedTemporaryFile(delete=False).name+".pypelid"

	# generate distribution of points on the sky
	index = np.arange(n)
	ra, dec = sphere.sample_sphere(n)
	skycoord = np.transpose([ra, dec])
	redshift = np.random.uniform(1., 2., n)

	# define metadata
	data = {
			'index': index,
			'skycoord': skycoord,
			'redshift': redshift,
			}
	dtypes = [
			np.dtype([('index', int, 1)]),
			np.dtype([('skycoord', float, 2)]),
			np.dtype([('redshift', float, 1)])
			]
	units = {'index': 'none', 'skycoord': 'degree', 'redshift': 'redshift'}
	meta = {'coordsys': 'equatorial J2000'}
	description = {'index': 'Pypelid catalogue index', 
				   'skycoord': 'RA and Dec coordinates', 
				   'redshift': 'Cosmological redshift'}

	# Load first
	with catalogue_store.CatalogueStore(filename, 'w', name='test',
		zone_resolution=zone_resolution, preallocate_file=False) as cat:
		cat.append(data)
		cat.load_attributes(test='ciao')
		cat.load_attributes(meta)
		cat.load_units(units)
		cat.load_description(description)
		assert(cat._datastore is not None)

		# Update second by setting 50% of the redshift values to 0
		p_update = 0.5
		index_update = np.random.choice([False,True], n, p=[1-p_update,p_update], replace=True)
		data_up = cat.to_structured_array()
		data_up['redshift'][index_update] = 0.0
		cat.update(data_up)

	# Compute check sums
	count = 0
	check_lon = 0
	check_lat = 0
	check_old_z = []
	check_new_z = []
	with catalogue_store.CatalogueStore(filename) as cat:
		print "count:", cat.count
		assert(cat.test == 'ciao')
		print "zones:", len(cat.get_zones())
		for group in cat:
			lon, lat = np.transpose(group.skycoord)
			count += len(lon)
			check_lon += np.sum(lon)
			check_lat += np.sum(lat)
			check_old_z.extend(group.redshift[group.redshift[:]>0])
			check_new_z.extend(group.redshift[group.redshift[:]==0])
	assert(count == n)
	assert(len(check_new_z)+len(check_old_z) == n)
	assert(np.allclose( check_lon, np.sum(skycoord[:, 0]) ))
	assert(np.allclose( check_lat, np.sum(skycoord[:, 1]) ))

	redshift = data_up['redshift']
	assert(np.allclose( np.sort(check_old_z), np.sort(redshift[np.logical_not(index_update)]) ))

	# delete the test file
	os.unlink(filename)

if __name__=='__main__':
	check_catalogue_store(n=10, zone_resolution=0)



def check_catalogue_store_batches(n=100, zone_resolution=0):
	""" Quick demo of catalogue store.

	This routine loads the data in batches.

	Generate mock data and write a catalogue store file.  Open the file and check
	that contents are correct.
	"""
	# temporary file to use for test
	filename = tempfile.NamedTemporaryFile(delete=False).name+".pypelid"

	# generate distribution of points on the sky
	ra, dec = sphere.sample_sphere(n)
	skycoord = np.transpose([ra, dec])
	redshift = np.random.uniform(1., 2., n)

	# define metadata
	data = {
			'skycoord': skycoord,
			'redshift': redshift,
			}
	dtypes = [
			np.dtype([('skycoord', float, 2)]),
			np.dtype([('redshift', float, 1)])
			]
	units = {'skycoord': 'degree', 'redshift': 'redshift'}
	meta = {'coordsys': 'equatorial J2000'}
	description = {'skycoord': 'RA and Dec coordinates', 'redshift': 'Cosmological redshift'}

	batch = max(1, len(ra)//10)

	with catalogue_store.CatalogueStore(filename, 'w', name='test',
		zone_resolution=zone_resolution, preallocate_file=True) as cat:

		print cat.columns

		# load in the catalogue in batches
		i = 0
		while i < len(ra):
			cat.preprocess(ra[i:i+batch], dec[i:i+batch])
			i += batch

		cat.allocate(dtypes)
		cat.append(data)
		cat.load_attributes(meta)
		cat.load_units(units)
		cat.load_description(description)

	# Compute check sums
	count = 0
	check_lon = 0
	check_lat = 0
	with catalogue_store.CatalogueStore(filename) as cat:
		columns = cat.columns
		assert(len(columns)==2)
		assert 'skycoord' in columns
		assert 'redshift' in columns

		print "zones:", len(cat)
		for group in cat:
			lon, lat = np.transpose(group.skycoord)
			count += len(lon)
			check_lon += np.sum(lon)
			check_lat += np.sum(lat)
	assert(count == n)
	assert(np.allclose(check_lon, np.sum(skycoord[:, 0])))
	assert(np.allclose(check_lat, np.sum(skycoord[:, 1])))

	# delete the test file
	os.unlink(filename)


def test_catalogue_store_1():
	""" Check that attempting to open a file that does not exist results in IOError
	"""
	filename = tempfile.NamedTemporaryFile(delete=False).name+".pypelid"

	try:
		catalogue_store.CatalogueStore(filename, mode='r')
	except IOError:
		pass

def test_catalogue_store_2():
	""" Check various dataset sizes and resolutions """
	cases = [(10000, 0), (1000, 0), (1000, 2), (1, 0)]
	for n, r in cases:
		yield check_catalogue_store, n, r
		yield check_catalogue_store_batches, n, r


def test_catalogue_store_3():
	""" Ensure that invalid resolution results in a ValueError. """
	try:
		check_catalogue_store(1, -1)
	except ValueError:
		pass

	try:
		check_catalogue_store_batches(1, -1)
	except ValueError:
		pass
