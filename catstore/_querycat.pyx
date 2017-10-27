# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import logging
import numpy as np
from sklearn.neighbors import KDTree

cimport numpy as np
cimport libc.math as math

DEG_TO_RAD = np.pi / 180.


cdef class QueryCat(object):
	""" """
	cdef double [:,:] xy
	cdef object _lookup_tree

	def __init__(self, double [:,:] xy):
		"""
		"""
		self.xy = xy
		self._lookup_tree = KDTree(xy)

	def query_disk(self, double [:,:] centers, double [:] radius):
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
		return self._lookup_tree.query_radius(centers, radius)

	def query_box(self,  double[:,:] centers, double[:] width, double[:] height, double pad_x=0.0, double pad_y=0.0, double orientation=0.):
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
		cdef long i, j, k, n, m, c
		cdef object[:] matches
		cdef long[:] match
		cdef double dx, dy, dxt, dyt, w, h, cx, cy
		cdef double [:] data_x
		cdef double [:] data_y
		cdef double theta = orientation * DEG_TO_RAD
		cdef double costheta = math.cos(theta)
		cdef double sintheta = math.sin(theta)

		cdef double [:,:] data_xy = self.xy

		n = width.shape[0]
		m = data_xy.shape[0]

		cdef double[:] r = np.zeros(n, dtype=float)
		cdef long[:] select = np.zeros(m, dtype=int)
		cdef long[:] s

		for i in range(n):
			r[i] = math.sqrt(width[i]*width[i] + height[i]*height[i]) / 2.

		matches = self.query_disk(centers, r)

		cdef object[:] results = np.zeros(n, dtype=object)

		for i in range(n):

			match = matches[i]
			m = match.shape[0]
			
			w = width[i]/2. + pad_x
			h = height[i]/2. + pad_y

			cx = centers[i, 0]
			cy = centers[i, 1]

			c = 0

			for j in range(m):

				k = match[j]
				dx = data_xy[k, 0] - cx
				dy = data_xy[k, 1] - cy

				dxt = dx * costheta - dy * sintheta
				dyt = dx * sintheta + dy * costheta

				if (math.fabs(dxt) < w) and (math.fabs(dyt) < h):
					select[c] = k
					c += 1

			s = np.zeros(c, dtype=int)
			for j in range(c):
				s[j] = select[j]

			results[i] = s

		return results
