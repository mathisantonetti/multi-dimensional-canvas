import manifold_detection.tools.manifolds
import manifold_detection.tools.std_grids
from manifold_detection.tools.basics import *
import copy
import numpy as np

def cube_hull(X):
    d = X.shape[1]
    return [np.min(X[:, i]) for i in range(d)], [np.max(X[:,i]) for i in range(d)]

def works(model, eval_inf, X):
    N, d = X.shape
    Id1 = [] # points working
    Id0 = [] # points not working
    for i in range(N):
        if(model(X[i, :]) > eval_inf):
            Id1.append(i)
        else:
            Id0.append(i)

    return Id1, Id0

def SeparateAndCut(grid, Id1, Id0, errII_m):
    grid_points = grid.points
    n0 = len(Id0)
    n1 = len(Id1)
    dim = grid_points.shape[1]
    globalparams = []
    globalId0_1 = []
    globalId0_2 = []
    globalId1_1 = []
    globalId1_2 = []
    global_err1 = 0.0
    global_err2 = 0.0
    for i in range(n0):
        for d in range(dim):
            # separation
            sepgrid_points1 = np.copy(grid_points[:n1, :])
            sepgrid_points2 = np.copy(grid_points[:n1, :])
            nsep1, nsep2 = 0, 0
            Id1_1 = []
            Id1_2 = []
            for j in range(n1):
                if(grid_points[Id1[j], d] > grid_points[Id0[i], d]):
                    nsep1 += 1
                    sepgrid_points1[nsep1-1, :] = np.copy(grid_points[Id1[j], :])
                    Id1_1.append(Id1[j])
                else:
                    nsep2 += 1
                    sepgrid_points2[nsep2-1, :] = np.copy(grid_points[Id1[j], :])
                    Id1_2.append(Id1[j])
            sepgrid_points1 = sepgrid_points1[:nsep1, :]
            sepgrid_points2 = sepgrid_points2[:nsep2, :]

            # domain decomposition
            params_domain1 = [[-np.Inf for k in range(dim)], [np.Inf for k in range(dim)]]
            params_domain2 = [[-np.Inf for k in range(dim)], [np.Inf for k in range(dim)]]
            if(nsep1 >= 1):
                params_domain1 = cube_hull(sepgrid_points1)
            if(nsep2 >= 1):
                params_domain2 = cube_hull(sepgrid_points2)

            # error estimate
            Id0_1 = [] # Id0 for domain 1
            cube1 = manifold_detection.tools.manifolds.Cube(params_domain1[0], params_domain1[1])
            Id0_2 = [] # Id0 for domain 2
            cube2 = manifold_detection.tools.manifolds.Cube(params_domain2[0], params_domain2[1])
            for j in range(n0):
                if(cube1(grid_points[Id0[j], :]) > 0.5):
                    Id0_1.append(Id0[j])
                if(cube2(grid_points[Id0[j], :]) > 0.5):
                    Id0_2.append(Id0[j])
            #print(len(Id1_1), " , ", len(Id0_1))

            local_err1 = div(len(Id1_1),len(Id0_1))

            local_err2 = div(len(Id1_2),len(Id0_2))

            # replacement
            if(min(global_err1, global_err2) < min(local_err1, local_err2)):
                globalparams = [copy.deepcopy(params_domain1), copy.deepcopy(params_domain2)]
                globalId0_1 = copy.deepcopy(Id0_1)
                globalId0_2 = copy.deepcopy(Id0_2)
                globalId1_1 = copy.deepcopy(Id1_1)
                globalId1_2 = copy.deepcopy(Id1_2)
                global_err1 = local_err1
                global_err2 = local_err2

    if(1/(1+div(len(globalId1_1),len(globalId0_1))) > errII_m):
        print(len(globalId1_1), " , ", len(globalId0_1))
        newparams, newerror1, newerror2 = SeparateAndCut(grid, globalId1_1, globalId0_1, errII_m)
        globalparams = [newparams[k] for k in range(len(newparams))] + [globalparams[1]]
        global_err1 = max(global_err1, min(newerror1, newerror2))

    print(len(globalId1_2), " , ", len(globalId0_2))
    if(1/(1+div(len(globalId1_2),len(globalId0_2))) > errII_m):
        newparams, newerror1, newerror2 = SeparateAndCut(grid, globalId1_2, globalId0_2, errII_m)
        globalparams = [globalparams[0]] + [newparams[k] for k in range(len(newparams))]
        global_err2 = max(global_err2, min(newerror1, newerror2))

    return globalparams, global_err1, global_err2

def Bayesian_estimate(points, Id1, Id0):
    pass

def SeparateAndExplore(grid, Id0, a, b, error_max=0.05, number_manifold_limit=100, _isfirst=True):
    dim = len(a)
    if(len(Id0) == 0):
        #print(a, " , , ", b)
        return [(a,b)]
    else:
        dim = grid.points[0].shape[0]
        median_ind, separation_inds = grid.median(indices=Id0, return_indices=True)
        median_point = grid.points[median_ind]

        propositions = []
        for dir in range(dim):
            new_a, new_b = [a[k] for k in range(dim)], [b[k] for k in range(dim)]
            new_a[dir] = median_point[dir]
            new_b[dir] = median_point[dir]
            propositions += SeparateAndExplore(grid, separation_inds[2*dir], new_a, b, error_max=error_max, _isfirst=False)
            propositions += SeparateAndExplore(grid, separation_inds[2*dir+1], a, new_b, error_max=error_max, _isfirst=False)

        if(_isfirst):
            sols = manifold_detection.tools.manifolds.Atlas([])
            volmax = np.inf
            props = [manifold_detection.tools.manifolds.Cube(propositions[i][0], propositions[i][1]) for i in range(len(propositions))]
            for k in range(number_manifold_limit):
                prop_indices = np.argsort([props[i].measure() - (sols.intersect(props[i])).measure() for i in range(len(props))])
                #print(sols.intersect(props[prop_indices[-1]]))
                if(k == 0):
                    volmax = props[prop_indices[-1]].measure() - (sols.intersect(props[prop_indices[-1]])).measure()
                elif((props[prop_indices[-1]].measure() - (sols.intersect(props[prop_indices[-1]])).measure())/volmax < error_max):
                    break
                sols.update([props[prop_indices[-1]]])

            return sols
        else:
            return propositions