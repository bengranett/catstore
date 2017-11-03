import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from catstore import catalogue


def test_query_disk(nqueries=1000, n=10000):
    imagecoord = np.random.uniform(0, 100, (n, 2))
    data = {'imagecoord': imagecoord}
    cat = catalogue.Catalogue(data)

    x,y = np.random.uniform(0,100,(2,nqueries))
    rad = np.random.uniform(1,50, nqueries)

    results = cat.query_disk(x, y, radius=rad)

    for i,matches in enumerate(results):
        if len(matches) == 0:
            continue
        center = np.array([x[i],y[i]])
        xy = data['imagecoord'][matches]
        r = np.sum((xy-center)**2, axis=1)**.5
        assert r.max() < rad[i]

def test_query_box(nqueries=100, nangles=20, n=10000):
    imagecoord = np.random.uniform(0, 100, (n, 2))
    data = {'imagecoord': imagecoord}
    cat = catalogue.Catalogue(data)

    for angle in np.linspace(0, 360, nangles):
        
        sintheta = np.sin(angle*np.pi/180)
        costheta = np.cos(angle*np.pi/180)

        x, y = np.random.uniform(0, 100,(2, nqueries))
        width = np.random.uniform(1, 5, nqueries)
        height = np.random.uniform(1, 10, nqueries)

        results = cat.query_box(x, y, width, height, pad_x=0, pad_y=0, orientation=angle)

        for i, matches in enumerate(results):
            if len(matches) == 0:
                continue

            matches = np.array(matches)

            xy = data['imagecoord'][matches]
            dx = xy[:,0] - x[i]
            dy = xy[:,1] - y[i]


            dxt = dx * costheta - dy * sintheta
            dyt = dx * sintheta + dy * costheta

            dxt = np.abs(dxt).max()
            dyt = np.abs(dyt).max()


            assert dxt < width[i]/2.
            assert dyt < height[i]/2.

def test_query_box_scaling(nqueries=100, nangles=13., n=100000):
    import matplotlib
    matplotlib.use('TKAgg')
    from matplotlib import pyplot

    imagecoord = np.random.uniform(0, 100, (n, 2))
    data = {'imagecoord': imagecoord}
    cat = catalogue.Catalogue(data, scale_x=1, scale_y=2, angle=30)

    for angle in np.linspace(0, 360, nangles):
        print "angle",angle
        sintheta = np.sin(angle*np.pi/180)
        costheta = np.cos(angle*np.pi/180)

        #x, y = np.random.uniform(0, 100,(2, nqueries))
        x, y = np.array([50., ]), np.array([50., ])
        width = np.ones(len(x))*40.0
        height = np.ones(len(x))*20.0

        results = cat.query_box(x, y, width, height, pad_x=0, pad_y=0, orientation=angle)


        for i, matches in enumerate(results):
            if len(matches) == 0:
                print "no match", x[i],y[i]
                continue

            matches = np.array(matches)

            xy = data['imagecoord'][matches]
            dx = xy[:,0] - x[i]
            dy = xy[:,1] - y[i]


            dxt = dx * costheta - dy * sintheta
            dyt = dx * sintheta + dy * costheta

            print len(dx)

            # pyplot.subplot(111,aspect='equal')

            # pyplot.plot(*imagecoord.T, marker=",", linestyle="None")

            # pyplot.plot(*xy.T, marker=".", linestyle="None")
            # xx = np.array([-width[0]/2, width[0]/2, width[0]/2, -width[0]/2, -width[0]/2.])
            # yy = np.array([-height[0]/2, -height[0]/2, height[0]/2, height[0]/2, - height[0]/2.])
            # xxt = xx * costheta + yy * sintheta
            # yyt = - xx * sintheta + yy * costheta

            # pyplot.plot(50+xxt, 50+yyt)

            # pyplot.show()

            dxt = np.abs(dxt).max()
            dyt = np.abs(dyt).max()

            print angle, dxt, width[i]/2, dyt, height[i]/2
            assert dxt < width[i]/2.
            assert dyt < height[i]/2.
# test_query_disk()
# test_query_box_scaling()
