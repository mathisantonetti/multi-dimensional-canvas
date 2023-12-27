import tools.manifolds
import tools.std_grids
from tools.basics import *
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
            cube1 = tools.manifolds.Cube(params_domain1[0], params_domain1[1])
            Id0_2 = [] # Id0 for domain 2
            cube2 = tools.manifolds.Cube(params_domain2[0], params_domain2[1])
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

def SeparateAndExplore(grid, Id0, a, b, number_manifolds):
    dim = len(a)
    if(len(Id0) == 0):
        #print(a, " , , ", b)
        return [(a,b)]
    else:
        dim = grid.points[0].shape[0]
        median_ind, separation_inds = grid.median(indices=Id0, return_indices=True)
        median_point = grid.points[median_ind]

        propositions = [(a,a)]*number_manifolds
        for dir in range(dim):
            new_a, new_b = [a[k] for k in range(dim)], [b[k] for k in range(dim)]
            new_a[dir] = median_point[dir]
            new_b[dir] = median_point[dir]
            explorations_high = SeparateAndExplore(grid, separation_inds[2*dir], new_a, b, number_manifolds)
            explorations_low = SeparateAndExplore(grid, separation_inds[2*dir+1], a, new_b, number_manifolds)

            vols = []
            vols_high = []
            vols_low = []
            for i in range(len(propositions)):
                vols.append(np.prod([propositions[i][1][k] - propositions[i][0][k] for k in range(dim)]))

            for i in range(len(explorations_high)):
                vols_high.append(np.prod([explorations_high[i][1][k] - explorations_high[i][0][k] for k in range(dim)]))

            for i in range(len(explorations_low)):
                vols_low.append(np.prod([explorations_low[i][1][k] - explorations_low[i][0][k] for k in range(dim)]))

            sorted_props = propositions + explorations_high + explorations_low
            sorted_indices = np.argsort(np.concatenate((vols, vols_high, vols_low)))
            propositions = [sorted_props[sorted_indices[-k-1]] for k in range(number_manifolds)]


        return propositions




model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 100)
grid.show(model, 0.5)


Id1, Id0 = grid.works(model, 0.5)
print(grid.points[grid.median(indices=Id0)])
#params, errIIsep1, errIIsep2 = SeparateAndCut(grid, Id1, Id0, 0.0)
params = SeparateAndExplore(grid, Id0, [0.0, 0.0], [1.0, 1.0], 3)

print(params)