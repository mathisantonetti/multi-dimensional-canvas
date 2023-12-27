import manifold_detection.tools as tools

def median_test(i):
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 100)
        grid.show(model, 0.5)


        Id1, Id0 = grid.works(model, 0.5)
        index, list_inds = grid.median(indices=Id0, return_indices=True)

        print(grid.points[index])
        print(list_inds)