import manifold_detection.tools as tools
import matplotlib.pyplot as plt
import numpy as np

def AdaManifPOST(a, b, alg, model, N):
    grid = tools.std_grids.CenteredGrid(a, b, N, 10, model=model, inv_detection=True)
    Id0 = np.where(np.array(grid.evals) < 0.5)[0]
    #print(Id0)
    params = alg(grid, Id0, a, b)
    n = N-grid.N
    while(n > 0):
        new_grid = tools.std_grids.CenteredGrid(params.manifolds[0].a, params.manifolds[0].b, n, 10, model=model, inv_detection=True)
        new_Id0 = np.where(np.array(new_grid.evals) < 0.5)[0]
        Id0 = np.append(Id0, [grid.N + i for i in new_Id0]).astype(int)
        #print(grid.points)
        grid.add(new_grid.points)
        #print(Id0)
        params = alg(grid, Id0, a, b)
        n = n - new_grid.N
    return params, grid