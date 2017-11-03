#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: profile=True

import logging
import numpy as np
from sklearn.neighbors import KDTree

from cython.view cimport array as cvarray
cimport numpy as np
cimport libc.math as math

DEF DEG_TO_RAD = 3.141592653589793 / 180.

cdef inline void rotate(double *x, double *y, double costheta, double sintheta):
	""" Rotate a coordinate pair in 2 dimensions around the origin. """
	cdef double xt = x[0] * costheta - y[0] * sintheta
	y[0] = x[0] * sintheta + y[0] * costheta
	x[0] = xt

cdef class QueryCat:
	""" """
	cdef double [:,:] xy
	cdef double [:,:] xyt
	cdef object _lookup_tree
	cdef double scale_x, scale_y
	cdef double sintheta, costheta

	def __init__(self, double [:,:] xy, double scale_x=1, double scale_y=1, double angle=0.):
		"""
		"""
		cdef double norm = math.sqrt(scale_x * scale_x / 2. + scale_y * scale_y / 2.)
		self.scale_x = scale_x / norm
		self.scale_y = scale_y / norm

		self.sintheta = math.sin(angle * DEG_TO_RAD)
		self.costheta = math.cos(angle * DEG_TO_RAD)

		self.xy = xy
		self.xyt = self.transform(xy)

		self._lookup_tree = KDTree(self.xyt)

	cdef double[:,:] transform(self, double[:,:] xy):
		""" """
		cdef int i

		cdef double[:,:] out = cvarray(
									shape=(xy.shape[0], xy.shape[1]),
									itemsize=sizeof(double),
									format='d'
								)

		for i in range(xy.shape[0]):
			out[i, 0] = xy[i, 0]
			out[i, 1] = xy[i, 1]
			rotate(&out[i,0], &out[i,1], self.costheta, self.sintheta)
			out[i, 0] *= self.scale_x
			out[i, 1] *= self.scale_y

		return out

	cpdef object[:] query_disk(self, double [:,:] centers, double [:] radius):
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
		cdef double[:, :] centers_t
		centers_t = self.transform(centers)
		return self._lookup_tree.query_radius(centers_t, radius)

	cpdef query_box(self,  double[:,:] centers, double[:] width, double[:] height, double pad_x=0.0, double pad_y=0.0, double orientation=0.):
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
		cdef double dx, dy, cx, cy, w1, h1, w2, h2, a, b, eff
		cdef double theta = orientation * DEG_TO_RAD
		cdef double costheta = math.cos(theta)
		cdef double sintheta = math.sin(theta)

		cdef double [:,:] data_xy = self.xy

		n = width.shape[0]
		m = data_xy.shape[0]

		cdef double[:] r = cvarray(shape=(n,), itemsize=sizeof(double), format='d')
		cdef long[:] select = cvarray(shape=(m,), itemsize=sizeof(long), format='l')
		cdef long[:] s

		for i in range(n):
			w1 = width[i]
			h1 = height[i]

			w2 = width[i]
			h2 = -height[i]

			rotate(&w1,&h1,costheta,-sintheta)
			rotate(&w2,&h2,costheta,-sintheta)
			rotate(&w1,&h1,self.costheta,self.sintheta)
			rotate(&w2,&h2,self.costheta,self.sintheta)

			a = w1*w1*self.scale_x*self.scale_x + h1*h1*self.scale_y*self.scale_y
			b = w2*w2*self.scale_x*self.scale_x + h2*h2*self.scale_y*self.scale_y

			if a > b:
				r[i] = math.sqrt(a)/2.
			else:
				r[i] = math.sqrt(b)/2.

		matches = self.query_disk(centers, r)

		cdef object[:] results = np.zeros(n, dtype=object)

		cdef long count = 0
		cdef long count_tot = 0

		for i in range(n):

			match = matches[i]
			m = match.shape[0]
			
			w1 = width[i]/2. + pad_x
			h1 = height[i]/2. + pad_y

			cx = centers[i, 0]
			cy = centers[i, 1]

			c = 0

			for j in range(m):

				k = match[j]
				dx = data_xy[k, 0] - cx
				dy = data_xy[k, 1] - cy

				rotate(&dx, &dy, costheta, sintheta)

				if (math.fabs(dx) < w1) and (math.fabs(dy) < h1):
					select[c] = k
					c += 1
			count += c
			count_tot += m

			s = np.zeros(c, dtype=int)
			for j in range(c):
				s[j] = select[j]

			results[i] = s

		eff = <double> count / <double> count_tot

		return results, eff
