import manifold_detection.tools.manifolds
import manifold_detection.tools.std_grids
from manifold_detection.tools.basics import *
from scipy.spatial import Voronoi
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
            return manifold_detection.tools.manifolds.Atlas([], parameters=propositions, shape="Cube")
        else:
            return propositions

def Compute_LECH(points, a, b):
    '''
    Parameters :
    _ points : a list of points assumed to be sorted in x (first coordinate)
    _ a : left-bottomost point of the bounding hypercube
    _ b : right-topmost point of the bounding hypercube

    Returns :
    _ LECH : The largest empty corner hypercube
    '''
    # Step 0
    CL = points[:len(points)//2]
    CR = points[len(points)//2:]

    # Step 1
    if(len(CL) > 0):
        new_b, new_a = b, a
        new_b[0] = points[len(points)//2][0]
        new_a[0] = points[len(points)//2][0]
        res_left = Compute_LECH(points[:len(points)//2], a, new_b)
        res_right = Compute_LECH(points[len(points)//2:], new_a, b)

    CL = np.sort(CL)[::-1]
    CR = np.sort(CR)[::-1]

    # Step 2
    s = len(CL)
    bottom_mates = []
    for i in range(s):
        for j in range(bottom_mates[-1], len(CR)):
            if(CR[j][1] > CL[i]):
                bottom_mates.append(j)
                break

    # Step 3
    stack = [bottom_mates[0]]
    CRprime = [CR[j] for j in range(len(CR))]
    for i in range(1, s):
        stack.append(bottom_mates[i-1])
        if(bottom_mates[i] == bottom_mates[i-1]):
            continue

        for j in range(bottom_mates[i-1]+1, s):
            if(j == bottom_mates[i]):
                break

            while(CR[j][0] < CR[stack[-1]][0] and stack[-1] >= bottom_mates[i-1]):
                CRprime.pop(stack[-1])
                stack.pop()

            stack.append(j)

def NoForget(grid, sorted_Id0, side_HC, update_HC, side="left"):
    for side_hc in side_HC:
        a_side, b_side = side_hc

        ab_side = np.copy(a_side)
        if(side == "right"):
            ab_side = b_side

        for i in range((len(sorted_Id0)+1)//2):
            candidate_pt = grid.points[sorted_Id0[len(sorted_Id0)//2+i]]
            #print(ab_side, "  ,  ,  ", candidate_pt)
            if(ab_side[0] != candidate_pt[0]):
                break
            if((a_side <= candidate_pt).all() and (candidate_pt <= b_side).all()):
                update_HC.append(side_hc)

        for i in range(1, len(sorted_Id0)//2):
            candidate_pt = grid.points[sorted_Id0[len(sorted_Id0)//2-i]]
            if(ab_side[0] != candidate_pt[0]):
                break
            if((a_side <= candidate_pt).all() and (candidate_pt <= b_side).all()):
                update_HC.append(side_hc)

def compute_MLHC(grid, a, b, Id0=None, is_sorted=False, right_side=False):
    if(Id0 is None):
        Id0 = np.arange(len(grid.points))

    #print(Id0, " ,, ", a, b)
    if(len(Id0) == 0):
        return [[a,b]], []

    sorted_Id0 = None
    if(is_sorted):
        sorted_Id0 = Id0
    else:
        sorted_inds = np.argsort([grid.points[Id0[j]][0] for j in range(len(Id0))])
        sorted_Id0 = [Id0[sorted_inds[k]] for k in range(len(sorted_inds))]

    separation_point = grid.points[sorted_Id0[len(sorted_Id0)//2]]

    new_b, new_a = np.copy(b), np.copy(a)
    new_b[0] = separation_point[0]
    new_a[0] = separation_point[0]
    right_HC, other_HC1 = compute_MLHC(grid, a, new_b, Id0 = sorted_Id0[:len(sorted_Id0)//2], is_sorted=True, right_side=True)
    left_HC, other_HC2 = compute_MLHC(grid, new_a, b, Id0 = sorted_Id0[len(sorted_Id0)//2+1:], is_sorted=True, right_side=False)

    side_HC, other_HC = [], other_HC1 + other_HC2

    # Do not forget
    NoForget(grid, sorted_Id0, left_HC, other_HC)
    NoForget(grid, sorted_Id0, right_HC, other_HC, side="right")

    # Fusion of side hypercubes
    for left_hc in left_HC:
        for right_hc in right_HC:
            a_left, b_left = left_hc
            a_right, b_right = right_hc
            a_fused, b_fused = np.maximum(a_left, a_right), np.minimum(b_left, b_right)
            a_fused[0] = a_right[0]
            b_fused[0] = b_left[0]
            #print(a_left, " , ", a_right, " , ", b_left, " , ", b_right)
            #print(a_fused, " : : ", b_fused)
            if((a_fused >= b_fused).any() == False):
                if((right_side and b[0] == b_fused[0]) or (right_side == False and a[0] == a_fused[0])):
                    for dir in range(1,grid.dim):
                        b_cand, a_cand = np.copy(b_fused), np.copy(a_fused)

                    if(a_fused[dir] < separation_point[dir] and separation_point[dir] < b_fused[dir]):
                        b_cand[dir] = separation_point[dir]
                        a_cand[dir] = separation_point[dir]

                        if((right_side and b[0] == b_fused[0]) or (right_side == False and a[0] == a_fused[0])):
                            side_HC.append([a_cand,b_fused])
                            side_HC.append([a_fused, b_cand])
                        else:
                            other_HC.append([a_cand,b_fused])
                            other_HC.append([a_fused,b_cand])
                    else:
                        if((right_side and b[0] == b_fused[0]) or (right_side == False and a[0] == a_fused[0])):
                            side_HC.append([a_fused, b_fused])
                        else:
                            other_HC.append([a_fused, b_fused])

    return side_HC, other_HC

def MLHC(grid, a, b, Id0=None):
    side_HC, other_HC = compute_MLHC(grid, a, b, Id0=Id0)
    result = manifold_detection.tools.manifolds.Atlas([], parameters=side_HC+other_HC, shape="Cube")

    return result


def Bayesian_estimate(points, Id1, Id0):
    pass

def ExhaustiveDiminisher(grid, Id0):
    '''
    Parameters :
    _ grid : the grid
    _ Id0 : indices of the points to take into account

    Returns :
    _ candidates : the estimated vertices of the Voronoi diagram.
    _ nearest_pt : the list of the nearest points associated to the vertices of same index.
    '''
    candidates = []
    nearest_pt = []
    for i in range(len(Id0)):
        x = grid.points[Id0[i]]
        sorted_Id0 = np.argsort([np.sum(np.power(x - grid.points[Id0[j]], 2)) for j in range(len(Id0))])
        Neighbour = [grid.points[Id0[sorted_Id0[j]]] for j in range(3)]
        for j in range(3, len(Id0)):
            if(sorted_Id0[j] == i):
                continue

            y3 = grid.points[Id0[sorted_Id0[j]]]
            isNotNeighbour = False
            for n1 in range(1, len(Neighbour)):
                y1 = Neighbour[n1]
                for n2 in range(1, len(Neighbour)):
                    if(n2 == n1):
                        continue
                    y2 = Neighbour[n2]
                    if(np.dot(y3-x, y2-x) > 0 and np.dot(y3-x, y1-x) > 0 and (np.dot(y2 - x, y2 - y3) < 0 or np.dot(y1 - x, y1 - y3) < 0)):
                        isNotNeighbour = True
                        break
                if(isNotNeighbour):
                    break
            if(isNotNeighbour == False):
                Neighbour.append(y3)

        if(len(Neighbour) >= grid.dim+1):
            def cands(I):
                if(len(I) == grid.dim+1):
                    #sol = np.linalg.lstsq([Neighbour[I[k]] - Neighbour[I[k+1]] for k in range(len(I)-1)], [(np.sum(np.power(Neighbour[I[k]], 2)) - np.sum(np.power(Neighbour[I[k+1]], 2)))/2 for k in range(len(I)-1)], rcond=None)[0]
                    A = [Neighbour[I[k]] - Neighbour[I[k+1]] for k in range(len(I)-1)]
                    if(np.linalg.cond(A) > 10**(10)):
                        #print("doesn : ", [Neighbour[I[k]] for k in range(len(I))])
                        return []
                    else:
                        return [np.linalg.solve([Neighbour[I[k]] - Neighbour[I[k+1]] for k in range(len(I)-1)], [(np.sum(np.power(Neighbour[I[k]], 2)) - np.sum(np.power(Neighbour[I[k+1]], 2)))/2 for k in range(len(I)-1)])]
                else:
                    L = []
                    for i in range(I[-1]+1, len(Neighbour)-grid.dim+len(I)):
                        L += cands(I+[i])
                    return L
            new_cands = cands([0])
            candidates += new_cands
            nearest_pt += len(new_cands)*[x]
        else:
            print("Something is wrong with this.")
            print(x, " : : ", Neighbour)

    new_candidates = []
    new_nearest_pt = []
    #print(candidates)
    for i in range(len(candidates)):
        if(oneclose(candidates[i], new_candidates, 10**(-10))):
            continue

        for k, point in enumerate(grid.points):
            val = np.dot(candidates[i], point - nearest_pt[i]) -  ((np.sum(np.power(point, 2)) - np.sum(np.power(nearest_pt[i], 2)))/2)
            #print(candidates[i], " , ", val)
            if(val > 10**(-10)):
                 break
            elif(k == len(grid.points)-1):
                new_candidates.append(candidates[i])
                new_nearest_pt.append(nearest_pt[i])

    return new_candidates, new_nearest_pt

def MLHS(grid, a, b):
    points = np.array(grid.points)
    vor = Voronoi(points)
    radius = []
    max_c = None
    max_radius = 0.0
    for c in vor.vertices:
        if((c<a).any() or (c>b).any()):
            continue

        radius.append(np.min(np.linalg.norm(c[None, :] - points, axis=1)))
        if(radius[-1] > max_radius):
            max_c = c
            max_radius = radius[-1]

    #index = np.argmax(radius)
    return max_c, max_radius