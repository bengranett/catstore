import numpy as np
from catstore import catalogue


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

def test_query_box(nqueries=10, n=1000):
    imagecoord = np.random.uniform(0, 100, (n, 2))
    data = {'imagecoord': imagecoord}
    cat = catalogue.Catalogue(data)

    x, y = np.random.uniform(0, 100,(2, nqueries))
    width = np.random.uniform(0.5, 10, nqueries)
    height = np.random.uniform(0.5, 10, nqueries)
    # angle = np.random.uniform(0, 360, nqueries)

    print len(x),len(y),len(width),len(height)
    results = cat.query_box(x, y, width, height, pad_x=0, pad_y=0, orientation=0)

    for i,matches in enumerate(results):
        print matches
        if len(matches)==0:
            continue
        xy = data['imagecoord'][matches]
        print xy.shape
        dx = np.abs(xy[:,0] - x[i]).max()
        dy = np.abs(xy[:,1] - y[i]).max()

        print dx,dy, width[i]/2., height[i]/2.

        assert dx < width[i]/2.
        assert dy < height[i]/2.

test_query_box()

# test_disk()