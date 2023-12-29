import manifold_detection.tools as tools
from manifold_detection.aposteriori_search import *
from manifold_detection.adaptive_search import *

def median_test(i):
    if(i == 0):
        grid = tools.std_grids.CenteredGrid([0.0, 0.0], [1.0, 1.0],  10, 10)
        model = tools.manifolds.Cube([0.0, 0.0], [1.0, 1.0])
        grid.show(model, 0.5)

        index = grid.median(plans=np.array([[1.0, 1.0], [1.0, -1.0]]))
        print(grid.points[index])
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 100)
        grid.show(model, 0.5)


        Id1, Id0 = grid.works(model, 0.5)
        index, list_inds = grid.median(indices=Id0, return_indices=True)

        print(grid.points[index])
        print(list_inds)

def separation_test(i):
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 100)
        grid.show(model, 0.5)


        Id1, Id0 = grid.works(model, 0.5)
        print(grid.points[grid.median(indices=Id0)])
        #params, errIIsep1, errIIsep2 = SeparateAndCut(grid, Id1, Id0, 0.0)
        params = SeparateAndExplore(grid, Id0, [0.0, 0.0], [1.0, 1.0])

        print(params)

def centergrid_test(i):
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.CenteredGrid([0.0, 0.0], [1.0, 1.0], 1000, 10, model=model, inv_detection=True)
        grid.show(model, 0.5)


def ada_test(i):
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        params, grid = AdaManifPOST([0.0, 0.0], [1.0, 1.0], SeparateAndExplore, model, 1000)
        #print(grid.points.shape)
        #print(grid.N)
        grid.show(model, 0.5)
        print(params)
        print(grid.points)