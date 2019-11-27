"""spherical geometry utilities"""

import numpy as np


# degrees to radian conversions
c = np.pi/180
ic = 180/np.pi


def lonlat2xyz(lon,lat,r=1):
    """ """
    x = r*np.cos(lon*c)*np.cos(lat*c)
    y = r*np.sin(lon*c)*np.cos(lat*c)
    z = r*np.sin(lat*c)
    return x,y,z


def xyz2lonlat(x,y,z,norm=True,getr=False):
    """ """
    if norm:
        r = np.sqrt(x*x+y*y+z*z)
    else:
        r = np.ones(x.shape)
    lat = np.arcsin(z/r)*ic
    lon = np.arctan2(y,x)*ic
    if getr:
        return lon,lat,r
    return lon,lat


def rotate_xyz(x,y,z,angles=None,inverse=False):
    """ Rotate a set of vectors pointing in the direction x,y,z

    angles is a list of longitude and latitude angles to rotate by.
    First the longitude rotation is applied (about z axis), then the
    latitude angle (about y axis).
    """
    if angles==None:
        return x,y,z

    xyz = np.array([x,y,z])
    for dphi,dlon,dlat in angles:
        dphi*=c
        dlon*=c
        dlat*=c
        m0 = np.array([[1,0,0],
                      [0, np.cos(dphi),np.sin(dphi)],
                      [0, -np.sin(dphi), np.cos(dphi)]])

        m1 = np.array([[np.cos(dlon),-np.sin(dlon),0],
                      [np.sin(dlon), np.cos(dlon),0],
                      [0,0,1]])

        m2 = np.array([[np.cos(dlat),0,-np.sin(dlat)],
                      [0,1,0],
                      [np.sin(dlat), 0, np.cos(dlat)]])

        m = np.dot(np.dot(m1,m2),m0)

    if inverse:
        m = np.linalg.inv(m)

    xyz2 = np.dot(m,xyz)
    return xyz2


def rotate_lonlat(lon, lat, angles=[(0, 0, 0)], inverse=False):
    """ Rotate a set of longitude and latitude coordinate pairs.
    """
    xyz = np.array(lonlat2xyz(lon,lat))
    xyz2 = rotate_xyz(*xyz,angles=angles, inverse=inverse)
    return xyz2lonlat(*xyz2,getr=False)
