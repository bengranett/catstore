import numpy as np
from catstore import catalogue
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot

def test_query_disk(nqueries=1000, n=10000):
    imagecoord = np.random.uniform(0, 100, (n, 2))
    data = {'imagecoord': imagecoord}
    cat = catalogue.Catalogue(data)

    x,y = np.random.uniform(0,100,(2,nqueries))
    rad = np.random.uniform(1,50, nqueries)

    results = cat.query_disk(x, y, radius=rad)

    for i,matches in enumerate(results):
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

        for i,matches in enumerate(results):
            if len(matches)==0:
                continue

            xy = data['imagecoord'][matches]
            dx = xy[:,0] - x[i]
            dy = xy[:,1] - y[i]

            dxt = dx * costheta - dy * sintheta
            dyt = dx * sintheta + dy * costheta

            dxt = np.abs(dxt).max()
            dyt = np.abs(dyt).max()

            assert dxt < width[i]/2.
            assert dyt < height[i]/2.
