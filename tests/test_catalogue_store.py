import sys
import os
import numpy as np
import pypelid.sky.catalogue_store as catalogue_store
import pypelid.utils.sphere as sphere
import logging

logging.basicConfig(level=logging.INFO)


def check_catalogue_store(n=100, zone_resolution=0):
	""" Quick demo of catalogue store.

	Generate mock data and write a catalogue store file.  Open the file and check
	that contents are correct.
	"""
	# temporary file to use for test
	filename = os.tempnam() + ".pypelid"

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

	with catalogue_store.CatalogueStore(filename, 'w', name='test',
		zone_resolution=zone_resolution, preallocate_file=False) as cat:
		# cat.preload(ra, dec)
		# cat.allocate(dtypes)
		cat.update(data)
		cat.update_attributes(meta)
		cat.update_units(units)
		cat.update_description(description)

	# Compute check sums
	count = 0
	check_lon = 0
	check_lat = 0
	with catalogue_store.CatalogueStore(filename) as cat:
		print "zones:", len(cat.get_zones())
		for group in cat.get_data():
			lon, lat = np.transpose(group['skycoord'])
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
	filename = os.tempnam() + ".pypelid"
	try:
		catalogue_store.CatalogueStore(filename, mode='r')
	except IOError:
		pass

def test_catalogue_store_2():
	""" Check various dataset sizes and resolutions """
	cases = [(10000, 0), (1000, 0), (1000, 2), (1, 0)]
	for n, r in cases:
		yield check_catalogue_store, n, r


def test_catalogue_store_3():
	""" Ensure that invalid resolution results in a ValueError. """
	try:
		check_catalogue_store(1, -1)
	except ValueError:
		pass

check_catalogue_store(100,0)
