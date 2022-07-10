using Pkg
using Plots
using LinearAlgebra
using Convex
using SCS

######## Tools ########
# Logic Tools

function oneclose(e, L, eps)
    #=
    Parameters : 
    _ L : list of vectors (in matrix shape) 
    _ e : element possibly near to L
    _ eps : maximum accuracy tolerated of e with one element in L

    Returns : boolean
    if e is eps-close to L, returns true
    else returns false
    =#
    n = size(L, 1)
    if(n == 0)
        return false
    end
    for i=1:n
        if(norm(e - L[i, :]) > eps)
            return false
        end
    end
    return true
end

function updateBase1(i, limits)
    #=
    Parameters :
    _ i : integer vector that symbolize the number to update
    _ limits : list of integer 2D-vectors to symbolize the box of the base

    Returns :
    The number i+1 in the base notation defined by limits
    =#
    d = length(i)
    new_i = i
    for k=1:d
        if(new_i[k] <= limits[k][2] - 1)
            new_i[k] += 1
            break
        elseif(k <= d - 1)
            new_i[k] = limits[k][1]
        else
            return -1
        end
    end
    return new_i
end

function update_changes(list, elementToAdd, Compare_list, compared_element)
    #=
    Parameters :
    _ list : list of elements sorted with the order defined by Compare_list
    _ elemntsToAdd : the element to add to list if compared_element greater than one in list.
    _ Compare_list : list of real numbers defining the order of list
    _ compared_element : real number defining the order

    Returns :
    list updated with elemntToAdd (added to the right place defined by its order)
    =#
    if(length(Compare_list) != length(list[:, 1]))
        print("\n list : ", list)
        print("\n clist : ", Compare_list)
    end
    K = length(Compare_list) 
    new_list = copy(list)
    new_compare_list = copy(Compare_list)
    dim = length(elementToAdd)
    for k=1:(K-1)
        if(Compare_list[k] < compared_element && compared_element <= Compare_list[k+1])
            Listcat = cat(list[2:k, :], reshape(elementToAdd, 1, dim), dims=1) # add the element
            new_list = cat(Listcat, list[(k+1):K, :], dims=1)
            Comparecat = cat(Compare_list[2:k], [compared_element], dims=1)
            new_compare_list = cat(Comparecat, Compare_list[(k+1):K], dims=1)
            break
        end
    end
    if(Compare_list[K] < compared_element)
        new_list[1:(K-1), :] = copy(list[2:K, :])
        new_compare_list[1:(K-1)] = copy(Compare_list[2:K])
        new_list[K, :] = elementToAdd
        new_compare_list[K] = compared_element
    end
    return new_list, new_compare_list
end

function isIncludedInOneHC(H1, list_H2)
    #=
    Parameters :
    _ H1 : Hypercube to check (same format as for the cube_model)
    _ list_H2 : list to check (same format as the assembly_cube_model)

    Returns :
    True if H1 is (strictly) included in an element of list_H2, else false
    =#
    d = size(H1, 2)
    n = size(list_H2, 1)
    for i=1:n
        count = 0
        for j=1:d
            if(H1[j][1] <= (list_H2[i])[j][1] || (list_H2[i])[j][2] <= H1[j][2]) # if we can have x in H1 and not in the interior of list_H2[i]
                break
            end
            count += 1
        end
        if(count == d)
            return true
        end
    end
    return false
end

function decompo_base(n, d; base=10)
    #=
    Parameters :
    _ n : Number to return
    _ d : return format control
    _ base : base control

    Returns :
    [a1, ..., ad] such that n = a1 + a2*base + ... + ad*base^(d-1) (base decomposition)    
    =#
    res = zeros(Int, d)
    x0 = n
    for i=1:d
        q = floor(Int, x0/base^(d-i))
        res[i] = q
        x0 = x0 - q*base^(d-i)
    end
    return res
end

# Grid Tools

function regular_grid(a, b, N)
    #=
    Parameters :
    _ a : minimum boundaries of [a, b]
    _ b : maximum boundaries of [a, b]
    _ N : number of points per coordinates

    Returns :
    a centered regular grid X of N^d points (where d is the dimension of a)    
    =#
    dim = length(a)
    Y = zeros((N^dim, dim))
    c = LinRange(a[dim] + (b[dim]-a[dim])/(2N), b[dim] - (b[dim]-a[dim])/(2N), N)
    if(dim >= 2)
        Z = regular_grid(a[1:(dim - 1)], b[1:(dim-1)], N)
        for i=1:N^(dim - 1)
            for j=1:N
                Y[(i-1)*N+j, 1:(dim - 1)] = Z[i, 1:(dim - 1)]
                Y[(i-1)*N+j, dim] = c[j]
            end
        end
        return Y
    else
        for j=1:N
            Y[j, :] = [c[j]]
        end
        return Y
    end
end

function step_seq_center(Vn, s)
    #=
    Parameters :
    _ Vn : n-th term of the sequence Vs
    _ s : control of the sequence Vs 

    Returns :
    _ V(n+1) : (n+1)-th term of the sequence Vs    
    =#
    kn = Vn[1]
    jn = Vn[2]
    if(jn >= 3)
        return [kn+1, jn-2]
    else
        return [max(2kn+jn-s + 1, 1), s-1-abs(2kn+jn-s)]
    end
end

function center_line(a, b, V)
    #=
    Parameters :
    _ a : minimum boundaries of [a, b]
    _ b : maximum boundaries of [a, b]
    _ V : tuple (k, j) giving the decision for drawing the lines

    Returns :
    _ Matrix Line which is the grid corresponding to V
    _ int i - 1 : length of Line 
    =#
    k = V[1]
    j = V[2]
    d = length(a)
    N = d*2^((k-1)*d)*2^(j - 1)
    Line = zeros(N, d)
    i = 1
    for t=0:(2^(d*(k-1)) - 1)
        n = zeros(Int64, 0)
        r = 0
        t0 = t
        for l=1:d
            r = t0%(2^(k-1))
            t0 = floor(Int, (t0 - r)/(2^(k-1)))
            append!(n, 2r+1)
        end
        if(j == 1)
            Line[i, :] = Diagonal(n/(2^k))*(b - a) + a
            i += 1
        else
            for p=0:(2^(j - 2) - 1)
                for l=1:d
                    Line[i, :] = Diagonal(n/(2^k))*(b - a) + a
                    Line[i+1, :] = Line[i, :]
                    Line[i, l] = Line[i, l] + ((2p+1)/(2^(k + j - 1)))*(b[l] - a[l])
                    Line[i+1, l] = Line[i+1, l] - ((2p+1)/(2^(k + j - 1)))*(b[l] - a[l])
                    i += 2
                end
            end
        end
    end
    return Line[1:(i - 1), :], i - 1
end

function center_based_grid(a, b, N, s)
    #=
    Parameters :
    _ a : minimum boundaries of [a, b]
    _ b : maximum boundaries of [a, b]
    _ N : maximal number of points in the grid
    _ s : control of the grid sequence (represents the minimum distance between points of the grid)

    Returns :
    _ X : the center-based grid more and more precisely filling each hypercube of maximal size 
    =#
    current_V = [1, 1]
    j = 1
    X = zeros(Float64, 0)
    sc = ceil(Int, s)
    while(j <= N && current_V[2] >= 1)
        Line, L = center_line(a, b, current_V)
        #print("\n current line : ", Line)
        for i=1:L
            #print("\n line : ", reshape(Line[i, :], 1, dim))
            X = cat(X, reshape(Line[i, :], 1, dim), dims=1) # add the element
            j += 1
            if(j > N)
                break
            end
        end
        current_V = step_seq_center(current_V, sc)
    end
    return X
end

# Van Der Corput function
function Van_Der_Corput_g(b, n)
    #=
    Parameters :
    _ b : decoding base
    _ n : number to represent

    Returns :
    sum_i d_i b^(-i) where n = sum_i d_i b^(i-1)
    =#
    m = n
    dk = zeros(Int64, 0)
    while(m >= 2)
        r = m%b
        append!(dk, r)
        m = floor((m-r)/b)
    end
    append!(dk, 1)
    L = length(dk)
    return sum([dk[i]*float(b)^(-i) for i=1:L])
end

# Hammersley sequence diluted in [a[1], c[1]]x[a[2], c[2]]x...x[a[d], c[d]]
function Hammersley_grid(a, c, b, N)
    #=
    Parameters :
    _ a : minimum boundaries of [a, c]
    _ c : maximum boundaries of [a, c]
    _ b : base for Van Der Corput function : [b_1, ..., b_(d-1)] where pgcd(b_i, b_j) = 1 for all i not equal to j
    _ N : number of points

    Returns :
    _ x : Hammersley grid generated with b and N points    
    =#
    s = length(b) # = d - 1
    x = zeros((N, s+1))
    for i=1:N
        for j=1:s
            x[i, j] = (c[j] - a[j])*Van_Der_Corput_g(b[j], i) + a[j]
        end
        x[i, s+1] = (c[s+1] - a[s+1])*i/N + a[s+1]
    end
    return x
end

function LineArange(a, b, pos, n0, i)
    #=
    Parameters :
    _ a : minimum boundaries of [a, b]
    _ b : maximum boundaries of [a, b]
    _ pos : position of the hyperplan where the line is drawn
    _ n0 : number of points
    _ i : drawing dimension 

    Returns :
    _ Matrix Y : line generated with the given parameters    
    =#
    dim = length(a)
    Y = zeros((n0, dim))
    c = LinRange(a[i]+(b[i]-a[i])/(2n0), b[i]-(b[i]-a[i])/(2n0), n0)
    for j = 1:n0
        Y[j, 1:(i-1)] = pos[1:(i-1)]
        Y[j, i] = c[j] # we change only the dimension i
        Y[j, (i+1):dim] = pos[(i+1):dim]
    end

    return Y
end

# returns the cut (deduced from the model with condition model > ad_inf) of the line going through the dimension i
function cutline(model, params, ad_inf, line, a, b, n0, i)
    #=
    Parameters :
    _ model : used model
    _ params : used model parameters
    _ ad_inf : used model control (model > ad_inf)
    _ line : the position in the hyperplan to draw & cut the line
    _ a : minimum boundaries of [a, b]
    _ b : maximum boundaries of [a, b]
    _ n0 : number of points
    _ i : the dimension the line is going through

    Returns :
    _ Y : the line produced
    _ [m, M] : the cut 1D-region boundaries
    =#
    dim = length(line) + 1
    Y = zeros((n0, dim))
    c = LinRange(a[i]+(b[i]-a[i])/(2n0), b[i]-(b[i]-a[i])/(2n0), n0)
    count, countmax, m, M = 0, 0, a[i], b[i]
    for j = 1:n0
        Y[j, 1:(i-1)] = line[1:(i-1)]
        Y[j, i] = c[j] # we change only the dimension i
        Y[j, (i+1):dim] = line[i:(dim-1)]
        if(model(Y[j, :], params) <= ad_inf) # majority election
            if(count > countmax)
                M = c[j] - (b[i]-a[i])/(2n0)
                m = c[j - count] - (b[i]-a[i])/(2n0)
                countmax = count
            end
            count = 0
        else
            count += 1
        end
    end

    if(count > countmax)
        M = c[n0] + (b[i]-a[i])/(2n0)
        m = c[n0 - count + 1] - (b[i]-a[i])/(2n0)
        countmax = count
    end

    return Y, m, M
end

# fills the holes with 2^dim point in the hypercube generated by [[m[1], M[1]], ..., [m[d], M[d]], and repeats the operation until Nmax
function quad_fill(m, M, Nmax)
    #=
    Parameters :
    _ m : minimum boundaries of [m, M]
    _ M : maximum boundaries of [m, M]
    _ Nmax : maximal number of points

    Returns :
    _ X : a grid exploring with a regular grid 2 by 2 by ... and then with a regular grid 4 by 4 by ... etc... until it has Nmax points    
    =#
    X = regular_grid(m, M, 2)
    if(length(X[:, 1]) > Nmax)
        X = X[1:Nmax, :]
    end
    dim = length(m)
    I = floor(Int, log2(1 + Nmax*(1 - 1/2^dim))/dim)
    for i=1:I
        X = cat(X, regular_grid(m, M, 2^(i+1)), dims=1)
    end
    if(2^((I+1)*dim) - 1 > Nmax*(2^dim - 1)/2^dim)
        X = cat(X, regular_grid(m, M, 2^(I+2)), dims=1)[1:Nmax, :]
    end
    return X
end

# returns an adaptive grid generated with cutlines and quadfills with boundaries [a[1], b[1]]x...x[a[dim], b[dim]]
function adaptive_grid_lin(model, params, ad_inf, a, b, n0)
    #=
    Parameters :
    _ model : used model
    _ params : used model parameters
    _ ad_inf : used analysis control (criteria) (model > ad_inf)
    _ a : minimum boundaries of [a, b]
    _ b : maximum boundaries of [a, b]
    _ n0 : number of points per lines

    Returns :
    _ X : adaptive grid of (d*n0 + 2^d) points using the cutline strategy, where d is the dimension of the problem 
    =#
    dim = length(a)
    middle = ((a + b)/2)[1:(dim - 1)]
    X = zeros((n0*dim + 2^dim, dim))
    m = copy(a)
    M = copy(b)
    for d=1:dim
        X[(n0*(d-1)+1):(n0*d), :], m[d], M[d] = cutline(model, params, ad_inf, middle, a, b, n0, d)
        if(d <= dim - 1)
            middle[d] = (m[d] + M[d])/2
        end
    end
    X[(dim*n0+1):(dim*n0+2^dim), :] = quad_fill(m, M, 2^dim)
    return X, m, M
end

# returns X_ada depending on the particular strategy of the adaptive grid
function disc_return(X_ada, m, M, Kareas, K, true_K, full_var, returns_full)
    if(returns_full)
        return X_ada, m, M, Kareas, full_var
    else
        return X_ada, m[(K - true_K + 1):K, :], M[(K - true_K + 1):K, :]
    end
end

# Generates the adaptive grid assuming the model is a group of hypercubes.
#=
    - model : toymodel used to generate the grid, should be replaced later on by the actual black-box model.

    - params : toymodel parameters, should be replaced too once the test is done.

    - ad_inf : limit of interest, all values generated by the model inferior to ad_inf will be categorized as crashes.

    - a & b : limits of the experience search, for example a = fill(-1.0, d), b = fill(1.0, d) gives [-1, 1]^d.

    - nominal_dist : nominal distance between points when we input lines of parameters in the model.
    
    - Nmax : maximal number of experience points allowed. In practice, it is the number of simulations and thus the time allocated to the task.

    - K : maximal number of hypercubes to return 
=#
function adaptive_grid_cube(model, params, ad_inf, a, b, nominal_dist, Nmax, K; returns_full = false, explore_center = true)
    # Initialization
    dim = length(a)
    m = ones((K, dim))
    M = ones((K, dim))
    for k=1:K
        M[k, :] = b
        m[k, :] = b
    end

    Kareas = fill(-Inf, K)
    X_ada = zeros((0, dim))
    full_var = true
    true_K = 0

    # Base of lodes
    L_base = 1 + 2^dim + 2^(2*dim)
    X_base = reshape((a + b)/2.0, 1, dim)
    X_base = cat(X_base, quad_fill(a, b, L_base - 1), dims=1)
    
    if(explore_center == false)
        X_base = X_base[2:L_base, :]
        L_base = L_base - 1
    end

    # Processing adaptive grid...
    j = 1

    for i=1:L_base
        actual_parameters = [[[m[k,l], M[k,l]] for l=1:dim] for k=1:K]
        if(assembly_cube_model(X_base[i, :], actual_parameters) > 0.5)
            #print("base quad_fill number ", i, " is already investigated in theory : ", X_base[i, :], "\n")
        else
            if(oneclose(X_base[i, :], X_ada, 1e-7) == false) # if we don't already have that point
                X_ada = cat(X_ada, reshape(X_base[i, :], 1, dim), dims=1) # add the element
                j += 1
                if(j > Nmax)
                    return disc_return(X_ada, m, M, Kareas, K, true_K, full_var, returns_full)
                end
            end

            if(model(X_base[i, :], params) > ad_inf)

                mini = X_base[i, :]
                maxi = X_base[i, :]
                for d=1:dim
                    n0 = floor(Int, (b[d] - X_base[i, d])/nominal_dist) + 1
                    candidates_X = LineArange(X_base[i, :], b, X_base[i, :], n0, d)
                    k = 1
                    while(k <= n0)
                        if(oneclose(candidates_X[k, :], X_ada, 1e-7) == false) # if we don't already have that point
                            X_ada = cat(X_ada, reshape(candidates_X[k, :], 1, dim), dims=1) # add the element
                            j += 1
                            if(j > Nmax)
                                return disc_return(X_ada, m, M, Kareas, K, true_K, full_var, returns_full)
                            end
                        end
                        if(model(candidates_X[k, :], params) <= ad_inf)
                            full_var = false
                            break
                        end
                        k += 1
                    end
                    if(k >= 2)
                        maxi[d] = candidates_X[k-1, d]
                    end
                    n0 = floor(Int, (X_base[i, d] - a[d])/nominal_dist) + 1
                    candidates_X = reverse(LineArange(a, X_base[i, :], X_base[i, :], n0, d), dims=1)
                    k = 1
                    while(k <= n0)
                        if(oneclose(candidates_X[k, :], X_ada, 1e-7) == false) # if we don't already have that point
                            X_ada = cat(X_ada, reshape(candidates_X[k, :], 1, dim), dims=1) # add the element
                            j += 1
                            if(j > Nmax)
                                return disc_return(X_ada, m, M, Kareas, K, true_K, full_var, returns_full)
                            end
                        end
                        if(model(candidates_X[k, :], params) <= ad_inf)
                            full_var = false
                            break
                        end
                        k += 1
                    end
                    if(k >= 2)
                        mini[d] = candidates_X[k-1, d]
                    end
                end
                # Change the K hypercubes if necessary
                area = sum([log(maxi[l] - mini[l] + (maxi[l] == mini[l])*(1e-7)) for l=1:dim])
                M, newKareas = update_changes(M, maxi, Kareas, area)
                m, Kareas = update_changes(m, mini, Kareas, area)
                true_K = min(K, true_K + 1)
            else
                full_var = false
            end
        end
    end
    # If Nmax is not completely used, imbricate the process...
    if(Nmax - j >= true_K && true_K >= 1)
        n = floor(Int, (Nmax - j)/true_K)
        new_M = copy(M)
        new_m = copy(m)
        new_Kareas = fill(-Inf, K)
        for k=(K-true_K+1):K
            grid, candidate_m, candidate_M, candidate_Kareas, candidate_full_var = adaptive_grid_cube(model, params, ad_inf, m[k, :], M[k, :], nominal_dist, n, K, returns_full=true, explore_center=false)
            X_ada = cat(X_ada, grid, dims=1)
            if(candidate_full_var == false)
                for i=1:K
                    new_m, _ = update_changes(new_m, candidate_m[i, :], new_Kareas, candidate_Kareas[i])
                    new_M, new_Kareas = update_changes(new_M, candidate_M[i, :], new_Kareas, candidate_Kareas[i])
                end
            else
                new_m, _ = update_changes(new_m, m[k, :], new_Kareas, Kareas[k])
                new_M, new_Kareas = update_changes(new_M, M[k, :], new_Kareas, Kareas[k])
            end
        end
        m = new_m
        M = new_M
        Kareas = new_Kareas
    end
    return disc_return(X_ada, m, M, Kareas, K, true_K, full_var, returns_full)
end

# Toymodels

function _0_1model(x, params)
    return 1.0
end

# params = [center (Vector), interior radius (positive number), exterior radius (positive number)], example : params = [[0.5, 0.5], 0.4, 0.8]
function ball_model(x, params)
    return (norm(x - params[1]) < params[2]) || (norm(x - params[1]) > params[3])
end

# params = [[limit_1_inf, limit_1_sup], ..., [limit_d_inf, limit_d_sup]], example : params = [[0.0, 1.0], [0.0, 1.0]] gives the cube [0, 1]^2
function cube_model(x, params)
    d = length(x)
    for i=1:d
        if(params[i][1] > x[i] || x[i] > params[i][2]) # if x is not in the "cube"
            return 0.0 # False
        end
    end
    return 1.0 # True
end

# params = [params_cube_1, params_cube_2, ...], it is preferable to have no intersection between them, but it may still works with non empty intersections
function assembly_cube_model(x, params)
    n = size(params, 1)
    return sum([cube_model(x, params[i]) for i=1:n])
end

# Calculus tools

# logarithm adapted to eps (because it converges toward -infinity when x converges toward 0)
function invexp(x, eps)
    return log(eps)*(0 <= x <= eps) + log(x)*(x > eps)
end

# euclidian metric distance
function dist(x, y)
    #d = length(x)
    #return maximum([abs(x[i] - y[i]) for i=1:d])
    return norm(x - y)
end

# heterogeneous distance matrix
function distMathetero(X, W, s)
    nX = size(X, 1)
    nW = size(W, 1)
    Mat = zeros(nX, nW)
    for i=1:nX
        for j=1:nW
            Mat[i,j] = dist(X[i, :], W[j, :])^s
        end
    end
    return Mat
end

# homogeneous distance matrix
function distMathomo(X, s)
    return distMathetero(X, X, s)
end

# metric linear interpolation
function interpolation_lin(model, params, X, s)
    N = size(X, 1)
    beta = [model(X[i, :], params) for i=1:N]
    return distMathomo(X, s)\beta
end

# gaussian interpolation
function interpolation_gauss(model, params, X, sig)
    N = size(X, 1)
    beta = [model(X[i, :], params) for i=1:N]
    Mat = zeros((N, N))
    for i=1:N
        for j=1:N
            Mat[i, j] = exp(-dist(X[i, :], X[j, :])^2/(2sig^2))
        end
    end
    return Mat\beta
end

# returns f(x) = sum_i alpha_i d(x, X_i)^s with params = [alpha, X, s]
function f_metric_lin(x, params)
    alpha = params[1]
    X = params[2]
    s = params[3]
    Ntrain = size(X, 1)
    return sum([alpha[i]*dist(x, X[i, :])^s for i=1:Ntrain])
end

function f_gauss(x, params)
    alpha = params[1]
    X = params[2]
    sig = params[3]
    Ntrain = size(X, 1)
    return sum([alpha[i]*exp(-dist(x, X[i, :])^2/(2sig^2)) for i=1:Ntrain])
end

function cube_measurement(params)
    prod = 1
    d = length(params)
    for i=1:d
        prod *= (params[i][2] - params[i][1])
    end
    return prod
end

function cube_intersect(params1, params2)
    newparams = Vector{Vector{Float64}}()
    d = length(params1)
    for i=1:d
        a = max(params1[i][1], params2[i][1])
        b = min(params1[i][2], params2[i][2])
        if(a > b)
            append!(newparams, [[0.0, 0.0]])
        else
            append!(newparams, [[max(params1[i][1], params2[i][1]), min(params1[i][2], params2[i][2])]])
        end
    end
    return newparams
end

function assembly_cube_measurement(params)
    s = 0
    K = length(params)
    if(K == 0)
        return 0.0
    elseif(K == 1)
        return cube_measurement(params[1])
    else
        d = length(params[1])
        for i=1:K
            for j=1:d
                if(params[i][j][1] == params[i][j][2])
                    deleteat!(params, i)
                    return assembly_cube_measurement(params)
                end
            end
        end
        return assembly_cube_measurement(params[1:(K-1)]) - assembly_cube_measurement([cube_intersect(params[k], params[K]) for k=1:(K-1)]) + cube_measurement(params[K])
    end
end

"""
    LassoEN(Y,X,γ,λ)

Do Lasso (set γ>0,λ=0), ridge (set γ=0,λ>0) or elastic net regression (set γ>0,λ>0).


# Input
- `Y::Vector`:     T-vector with the response (dependent) variable
- `X::VecOrMat`:   TxK matrix of covariates (regressors)
- `γ::Number`:     penalty on sum(abs.(b))
- `λ::Number`:     penalty on sum(b.^2)

"""
function LassoEN(Y, X, γ, λ = 0)
    (T, K) = (size(X, 1), size(X, 2))

    b_ls = X \ Y                    #LS estimate of weights, no restrictions

    Q = X'X / T
    c = X'Y / T                      #c'b = Y'X*b

    b = Variable(K)              #define variables to optimize over
    L1 = quadform(b, Q)            #b'Q*b
    L2 = dot(c, b)                 #c'b
    L3 = norm(b, 1)                #sum(|b|)
    L4 = sumsquares(b)            #sum(b^2)

    if λ > 0
        Sol = minimize(L1 - 2 * L2 + γ * L3 + λ * L4)      #u'u/T + γ*sum(|b|) + λ*sum(b^2), where u = Y-Xb
    else
        Sol = minimize(L1 - 2 * L2 + γ * L3)               #u'u/T + γ*sum(|b|) where u = Y-Xb
    end
    solve!(Sol, SCS.Optimizer; silent_solver = true)
    Sol.status == Convex.MOI.OPTIMAL ? b_i = vec(evaluate(b)) : b_i = NaN

    return b_i, b_ls
end

function SeparateAndCut(model, params, grid, Id1, Id0, errII_m)
    #Id1, Id0 = works(model, params, eval_inf, grid)
    n0 = length(Id0)
    n1 = length(Id1)
    dim = length(grid[1, :])
    globalparams = zeros(Float32, 0)
    globalId0_1 = zeros(Int64, 0)
    globalId0_2 = zeros(Int64, 0)
    globalId1_1 = zeros(Int64, 0)
    globalId1_2 = zeros(Int64, 0)
    global_err1 = 0.0
    global_err2 = 0.0
    for i=1:n0
        for d=1:dim 
            # separation
            sepgrid1 = copy(grid[1:n1, :])
            sepgrid2 = copy(grid[1:n1, :])
            nsep1, nsep2 = 0, 0
            Id1_1 = zeros(Int64, 0)
            Id1_2 = zeros(Int64, 0)
            for j=1:n1
                if(grid[Id1[j], d] > grid[Id0[i], d])
                    nsep1 += 1
                    sepgrid1[nsep1, :] = copy(grid[Id1[j], :])
                    append!(Id1_1, Id1[j])
                else
                    nsep2 += 1
                    sepgrid2[nsep2, :] = copy(grid[Id1[j], :])
                    append!(Id1_2, Id1[j])
                end
            end
            sepgrid1 = sepgrid1[1:nsep1, :]
            sepgrid2 = sepgrid2[1:nsep2, :]

            # domain decomposition
            params_domain1 = [[-Inf, Inf] for k=1:dim]
            params_domain2 = [[-Inf, Inf] for k=1:dim]
            if(nsep1 >= 1)
                params_domain1 = cube_hull(sepgrid1)
            end
            if(nsep2 >= 1)
                params_domain2 = cube_hull(sepgrid2)
            end

            # error estimate
            Id0_1 = zeros(Int64, 0) # Id0 for domain 1
            Id0_2 = zeros(Int64, 0) # Id0 for domain 2
            for j=1:n0
                if(cube_model(grid[Id0[j], :], params_domain1) > 0.5)
                    append!(Id0_1, Id0[j])
                end
                if(cube_model(grid[Id0[j], :], params_domain2) > 0.5)
                    append!(Id0_2, Id0[j])
                end
            end
            local_err1 = length(Id1_1)/length(Id0_1)
            local_err2 = length(Id1_2)/length(Id0_2)
            
            # replacement
            if(min(global_err1, global_err2) < min(local_err1, local_err2))
                globalparams = [copy(params_domain1), copy(params_domain2)]
                globalId0_1 = copy(Id0_1)
                globalId0_2 = copy(Id0_2)
                globalId1_1 = copy(Id1_1)
                globalId1_2 = copy(Id1_2)
                global_err1 = local_err1
                global_err2 = local_err2
            end
        end
    end
    if(1/(1+(length(globalId1_1)/length(globalId0_1))) > errII_m)
        newparams, newerror1, newerror2 = SeparateAndCut(model, params, grid, globalId1_1, globalId0_1, errII_m)
        globalparams = cat([newparams[k] for k=1:length(newparams)], [globalparams[2]], dims=1)
        global_err1 = max(global_err1, min(newerror1, newerror2))
    end
    if(1/(1+(length(globalId1_2)/length(globalId0_2))) > errII_m)
        newparams, newerror1, newerror2 = SeparateAndCut(model, params, grid, globalId1_2, globalId0_2, errII_m)
        globalparams = cat([globalparams[1]], [newparams[k] for k=1:length(newparams)], dims=1)
        global_err2 = max(global_err2, min(newerror1, newerror2))
    end
    return globalparams, global_err1, global_err2
end

# Converting Tools
# Converts {Mat1, Mat2} to Vector{Vector}
function MatToVec(Mat1, Mat2)
    n, p = size(Mat1)
    return [[[Mat1[i, j], Mat2[i, j]] for j=1:p] for i=1:n]
end

# translates a List into a Tuple (for scatter use)
function ListToTuple(X)
    N = size(X, 1)
    newL = fill((1.0, 1.0), N)
    for i=1:N
        newL[i] = (X[i][1], X[i][2])
    end
    return newL
end

# separates points working and points not working (depending on the eval_inf (sensibility value))
function works(model, params, eval_inf, X)
    N = size(X, 1)
    d = length(X[1, :])
    Id1 = zeros(Int64, 0) # points working
    Id0 = zeros(Int64, 0) # points not working
    for i=1:N
        if(model(X[i, :], params) > eval_inf)
            append!(Id1, i)
        else
            append!(Id0, i)
        end
    end
    return Id1, Id0
end

# returns [label1, label2, ...], maxlabels, neighbours 
# such that for x_i = n + (-1)^(i+1)*N_test^(d - |_(i+1)/2_|), labeli = 0 if X[x_i] doesn't exist, and labeli = X_labels[x_i] otherwise
# maxlabels = max(nlabels)
# neighbours = [s1, s2, ...] where for all i, X_labels[si] = labeli
function labels_neighbourhood(n, X_labels, N_test, d)
    nlabels = zeros(Int64, 2d)
    max_label = 0
    for i=1:d
        xi = n + N_test^(d - i)
        if(xi <= N_test)
            nlabels[i] = X_labels[xi]
            if(nlabels[i] > max_label)
                max_label = n_labels[i]
            end
        end
        xi = n - N_test^(d - i)
        if(xi >= 1)
            nlabels[i+d] = X_labels[xi]
            if(nlabels[i+d] > max_label)
                max_label = nlabels[i+d]
            end
        end
    end
    return nlabels, max_label
end        

function compute_volume(n, X_labels, N_test, d)
    inds = zeros(Int, 1)
    true_N = length(X_labels)
    for k=1:d
        new_inds = (-N_test^(k-1)) .+ inds
        inds = cat(inds, new_inds, dims=1)
        new_N = length(new_inds)
        for j=1:new_N
            if(n + new_inds[j] < 1)
                return 0
            elseif(X_labels[n + new_inds[j]] == 0)
                return 0
            end
        end
    end
    return 1
end

function fuse_regions!(CRindexs, CRlabel_test_X, areas, sorted_labels, N_test, d, det_j)
    #=
    Parameters :
    _ CRindexs : list of indexs representing the regions
    _ CRlabel_test_X : labels of the points in test_X (the grid)
    _ areas : list of the area size for each region
    _ sorted_labels : sorted list of labels of the neighbourhood
    _ N_test : number of points per coordinate
    _ d : dimension of the problem vector space 
    _ det_j : first region to check

    Returns :
    Nothing
    =#
    n = length(sorted_labels)
    slabels_fus = sorted_labels[det_j]
    slabels_k = slabels_fus
    count_l = zeros(N_test^d)
    for k=(det_j+1):n
        slabels_k = sorted_labels[k]
        append!(CRindexs[slabels_fus], CRindexs[slabels_k]) # add region at i+1 to det_j
        areas[slabels_fus] += areas[slabels_k] # add volume
        for ind=1:N_test^d
            if(CRlabel_test_X[ind] == slabels_k)
                CRlabel_test_X[ind] = slabels_fus
            end
        end 
    end
    for k=(det_j+1):n
        slabels_k = sorted_labels[k]
        for ind=1:N_test^d
            if(CRlabel_test_X[ind] >= slabels_k+1)
                count_l[ind] += 1
            end
        end
    end
    CRlabel_test_X = CRlabel_test_X - count_l
    for k=(det_j+1):n
        slabels_k = sorted_labels[n + det_j + 1 - k]
        deleteat!(CRindexs, slabels_k) # remove region at k+1
        deleteat!(areas, slabels_k)
    end
end

# Returns the connected regions where forall x, model(x, params) > eval_inf
function AbstractToConnected(model, params, eval_inf, a, b, N_test)
    test_X = regular_grid(a, b, N_test)
    d = length(a)
    CRlabel_test_X = zeros(Int64, N_test^d)
    CRindexs = Vector{Vector{Int64}}()
    areas = zeros(Int, 0)
    for i = 1:N_test^d        
        nlabels, max_label = labels_neighbourhood(i, CRlabel_test_X, N_test, d)
        if(model(test_X[i, :], params) > eval_inf) # if test_X[i] is considered a good parameter
            if(max_label >= 1) # if there exists a neighbour s of test_X[i] in X such that CR_label_test_X[s] >= 1 
                sorted_labels = sort(unique(nlabels), alg=QuickSort)

                det_j = 2 # first region to check (second if we have 0 which represent failure)
                if(sorted_labels[1] >= 1)
                    det_j = 1
                end

                # fuse all regions in the neighbourhood
                fuse_regions!(CRindexs, CRlabel_test_X, areas, sorted_labels, N_test, d, det_j)
                
                # add the element to the region
                CRlabel_test_X[i] = sorted_labels[det_j]
                
                min_previous_n = minimum(nlabels[(d+1):(2d)])
                # add the element if it is on the region boundary
                if(min_previous_n == 0)
                    append!(CRindexs[sorted_labels[det_j]], i)
                end

                # volume calculus
                areas[sorted_labels[det_j]] += compute_volume(i, CRlabel_test_X, N_test, d)
            
            else # if there isn't any region in the neighbourhood, we add a new region
                append!(CRindexs, [[i]])
                append!(areas, 0)
                CRlabel_test_X[i] = length(CRindexs)
            end
        else
            # add the neighbours if it is on the region boundary
            for k=1:d
                if(nlabels[k+d] >= 1)
                    if((i - N_test^(d-k) in(CRindexs[nlabels[k+d]])) == false)
                        append!(CRindexs[nlabels[k+d]], i - N_test^(d-k))
                    end
                end
            end
        end
    end

    return test_X, CRindexs, CRlabel_test_X, areas
end

function ConnectedToHypercube(K, boundary_inds, labels, N_test, dim)
    #= 
    ---
    parameters :
    _ K : number max of hypercubes 
    _ boundary_inds : boundary points indexs (sorted) from a regular grid N_test^dim
    _ labels : labels of the grid points
    _ N_test : accuracy of the regular grid (number of points per line)
    _ dim : dimension of the elements of the grid
    ---
    returns the K maximal hypercubes found in the connected region
    =#
    N = length(boundary_inds)
    digit_BI = [reverse(decompo_base(boundary_inds[i] - 1, dim, base=N_test), dims=1) for i=1:N]
    Kareas = fill(-1, K)
    Kcubes_inds = zeros(Int, (K, 2))
    C_Hull = cube_hull(mapreduce(permutedims, vcat, digit_BI))
    print(C_Hull)
    j = [C_Hull[k][1]+1 for k=1:dim]
    while(j != -1)
        i = [C_Hull[k][1] for k=1:dim]
        while(i != -1)
            area = 1
            stop = false
            l = [i[k] for k=1:dim]
            while(l != -1)
                n = sum([l[k]*N_test^(k-1) for k=1:dim]) + 1
                if(labels[n] == 0)
                    stop = true
                    break
                end
                l = updateBase1(l, [[i[k]+1, j[k]-1] for k=1:dim])
            end
            for k=1:dim
                area = area*(j[k] - i[k])
            end
            if(stop == false)
                numi = sum([i[k]*N_test^(k-1) for k=1:dim]) + 1
                numj = sum([j[k]*N_test^(k-1) for k=1:dim]) + 1
                Kcubes_inds, Kareas = update_changes(Kcubes_inds, [numi, numj], Kareas, area)
                break
            end
            i = updateBase1(i, [[C_Hull[k][1], j[k] - 1] for k=1:dim])

        end
        j = updateBase1(j, C_Hull)
    end
    return Kcubes_inds, Kareas
end

function cube_hull(X)
    dim = length(X[1, :])
    return [[minimum(X[:, i]), maximum(X[:, i])] for i=1:dim]
end

function assembly_cube_hull(X)
    l = length(X)
    return [cube_hull(X[i]) for i=1:l]
end

# Canvas tools

# shows the grid X with the separation of good and bad points for the model
function showGrid(model, params, eval_inf, X)
    Idex1, Idex0 = works(model, params, eval_inf, X)
    tuple_real_X0 = ListToTuple([X[i, :] for i in Idex0])
    tuple_real_X1 = ListToTuple([X[i, :] for i in Idex1])
    scatter(tuple_real_X1, xlims = (-1, 1), ylims = (-1, 1), color = "blue", label = "success")
    scatter!(tuple_real_X0, xlims = (-1, 1), ylims = (-1, 1), color = "red", label = "fail")
end

# displays the abstract solution f based on the pre-computed solution alpha and the grid X
function display_sol(a, b, f, params)
    x = LinRange(a, b, 100)
    y = LinRange(a, b, 100)
    col = zeros(100, 100)
    for i=1:100
        for j=1:100
            col[j,i] = f([x[i], y[j]], params)
        end
    end
    heatmap(y, x, col)
end

# Validation Tools

# computes the type I and type II errors based on grid and the sensibility eval_inf
function success_rate(model, params, eval_inf, grid, val)
    n_test = size(grid, 1)
    count_type_I = 0
    count_type_II = 0
    count_reals = 0
    for i=1:n_test
        if(val[i] <= eval_inf)
            count_reals += 1
        end
        if(model(grid[i, :], params) > eval_inf)
            if(val[i] <= eval_inf) # It would success but we think it is a failure
                count_type_I += 1
            end
        else
            if(val[i] > eval_inf) # It would fail but we think it is a success
                count_type_II += 1
            end
        end
    end
    if(count_reals == 0)
        return 0.0, count_type_II/(n_test - count_reals)
    elif(n_test == count_reals)
        return count_type_I/count_reals, 0.0
    else
        return count_type_I/count_reals, count_type_II/(n_test - count_reals)
    end
end

function success_rate_assembly_cube(a, b, params_true, params_hat)
    K1 = length(params_true)
    K2 = length(params_hat)
    A = Vector{Vector{Vector{Float64}}}()
    d = length(a)
    for i=1:K1
        for j=1:K2
            append!(A, [cube_intersect(params_true[i], params_hat[j])])
        end
    end
    # print("A = ", A, "\n")
    # calculus lambda(U_{i=1}^{k1 k2} A_i) ...
    Dv_inter_Dhat = assembly_cube_measurement(A)
    Dv = assembly_cube_measurement(params_true)
    Dhat = assembly_cube_measurement(params_hat)
    D = cube_measurement([[a[k], b[k]] for k=1:d])
    if(Dhat == 0.0)
        return NaN, NaN
    else
        return (Dv - Dv_inter_Dhat)/(D - Dhat), 1.0 - (Dv_inter_Dhat/Dhat)
    end
end


# retrieve the model from metric interpolation with params = [alpha, grid, inter_inf] (where alpha is the interpolation restult, grid is the grid used to interpolate, and inter_inf is the sensibility)
function retrieve_metric_model(x, params)
    alpha = params[1]
    grid = params[2]
    s = params[3]
    interpolation_inf = params[4]
    N_train = size(grid, 1)
    return sum([alpha[k]*dist(x, grid[k, :])^s for k=1:N_train]) > interpolation_inf
end

function retrieve_gauss_model(x, params)
    alpha = params[1]
    grid = params[2]
    sig = params[3]
    interpolation_inf = params[4]
    N_train = size(grid, 1)
    return sum([alpha[k]*exp(-dist(x, grid[k, :])^2/(2sig^2)) for k=1:N_train]) > interpolation_inf
end

######## Tests ########
#=
# Hammersley test for metric interpolation on a group of cube model
begin
    dims = 2:6
    model_params = [[[0.0, 1.0], [0.0, 0.8], [0.1, 0.9], [0.3, 1.0], [-1.0, -0.2], [-1.0, 1.0]], [[-1.0, -0.4], [-0.9, -0.2], [0.4, 1.0], [-1.0, 1.0], [-0.9, -0.3], [0.7, 1.0]]]
    n_tests = [316, 46, 17, 10, 6]
    Ham_nums = [2, 3, 5, 7, 11]
    a_effs = LinRange(0.01, 0.99, 20)
    a = fill(-1.0, 6) # minimum boundaries
    b = fill(1.0, 6) # maximum boundaries
    errorsI = zeros(5, 20)
    errorsII = zeros(5, 20)

    for i=1:5
        dim = dims[i]

        # Change parameters depending on dimension
        model_params_dim = [model_params[1][1:dim], model_params[2][1:dim]]
        
        # Generates the training and validation grid
        Ham_X = Hammersley_grid(a[1:dim], b[1:dim], Ham_nums[1:(dim-1)], 1000)
        test_X = regular_grid(a[1:dim], b[1:dim], n_tests[i])
        print("\n succes grids, ")

        # solve interpolation
        alpha = interpolation_lin(assembly_cube_model, model_params_dim, Ham_X, 1)

        # validation for different values of a_effs
        #val = [f_metric_lin(test_X[i, :], [Ham_X, 1, alpha]) for i=1:n_tests[i]^dim]
        for j=1:20
            # No Projection
            #errorsI[i, j], errorsII[i, j] = success_rate(assembly_cube_model, model_params_dim, 1 - a_effs[j], test_X, val)
            #=
            # Big Projection
            print("n = ",n_tests[dim-1])
            CR_grid, CR_indexs, labels, areas = AbstractToConnected(retrieve_metric_model, [alpha, Ham_X, 1, 1 - a_effs[j]], 1 - a_effs[j], a[1:dim], b[1:dim], n_tests[dim-1])
            D_hat = assembly_cube_hull([CR_grid[CR_indexs[m], :] for m=1:length(CR_indexs)])
            D_v = model_params_dim
            errorsI[i, j], errorsII[i, j] = success_rate_assembly_cube(a, b, D_v, D_hat)
            =#
            #=
            # Sharp Projection
            CR_grid, CR_indexs, labels, areas = AbstractToConnected(retrieve_metric_model, [alpha, Ham_X, 1, 1 - a_effs[j]], 1 - a_effs[j], a[1:dim], b[1:dim], n_tests[dim-1])
            cubes, areas = ConnectedToHypercube(4, CR_indexs[ind], labels, n_tests[j], dim)
            TODO : Create the model generated
            =#
        end
    end
    plot(a_effs, [errorsI[1, :], errorsI[2, :], errorsI[3, :], errorsI[4, :], errorsI[5, :]], ylim=[0, 0.1], grid=false, w=[1 1], linestyle=[:solid :solid], label=["eI (d=2)" "eI (d=3)" "eI (d=4)" "eI (d=5)" "eI (d=6)"])
    #plot(a_effs, [a_effs, errorsII[1, :], errorsII[2, :], errorsII[3, :], errorsII[4, :], errorsII[5, :]], ylim=[0, 1], grid=false, w=[1 1], linestyle=[:dot :solid :solid :solid :solid :solid], label=["alpha_eff" "eII (d=2)" "eII (d=3)" "eII (d=4)" "eII (d=5)" "eII (d=6)"])
end
=#
#=
# Test for theory estimation consistency
begin
    # Parameters
    dim = 2
    model_params = [[[0.0, 1.0], [0.0, 0.8], [0.1, 0.9]], [[-1.0, -0.4], [-0.9, -0.2], [0.4, 1.0]]]
    n_test = 1000
    n0 = 10 # warning : no odd integer
    interpolation_inf = 0.8
    a = fill(-1.0, dim) # minimum boundaries
    b = fill(1.0, dim) # maximum boundaries
    K = 5 # maximal number of hypercubes to find (should be strictly greater than necessary)
    Ham_nums = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Change parameters depending on dimension
    model_params_dim = [model_params[1][1:dim], model_params[2][1:dim]]

    test_X = regular_grid(a, b, n_test)
    Ham_X = Hammersley_grid(a, b, Ham_nums[1:(dim-1)], 24)
    alpha = interpolation_lin(assembly_cube_model, model_params_dim, Ham_X, 1)

    CR_grid, CR_indexs, labels, areas = AbstractToConnected(retrieve_metric_model, [alpha, Ham_X, 1, 0.8], 0.8, a, b, 20)
    D_hat = assembly_cube_hull([CR_grid[CR_indexs[m], :] for m=1:length(CR_indexs)])

    D_v = model_params_dim
    err = success_rate_assembly_cube(a, b, D_v, D_hat)
    val = [assembly_cube_model(test_X[i, :], D_hat) for i=1:n_test^2]
    err2 = success_rate(assembly_cube_model, model_params_dim, 0.8, test_X, val)
    print(D_v, "\n")
    print(D_hat, "\n")
    print("error with theory : ", err)
    print("\n error with abstract : ", err2)
end
=#
# Adaptive Grids Test
#=
begin
    #model_params = [[[0.0, 1.0], [0.0, 0.8], [0.1, 0.9], [0.3, 1.0], [-1.0, -0.2], [-1.0, 1.0], [-1.0, -0.6], [0.4, 0.7], [-1.0, 1.0], [0.0, 1.0]], [[-1.0, -0.4], [-0.9, -0.2], [0.4, 1.0], [-1.0, 1.0], [-0.9, -0.3], [0.7, 1.0], [-1.0, -0.6], [-0.4, 0.1], [0.001, 0.5], [-0.6, -0.2]]] # modele difficile
    model_params = [[[0.0, 1.0], [0.0, 0.8], [0.1, 0.9], [0.3, 1.0], [-1.0, -0.2], [-1.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]], [[-1.0, -0.4], [-0.9, -0.2], [0.4, 1.0], [-1.0, 1.0], [-0.9, -0.3], [0.7, 1.0], [-0.8, -0.3], [0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]] # modele facile
    test_dists = LinRange(1e-3, 0.1, 100)
    a = fill(-1.0, 10) # minimum boundaries
    b = fill(1.0, 10) # maximum boundaries
    Kmax = 5
    errorsI = zeros(9, 100)
    errorsII = zeros(9, 100)
    
    for dim=2:10
        model_params_dim = [model_params[1][1:dim], model_params[2][1:dim]]

        for j=1:100
            adapt_X_cube, minim, Maxim = adaptive_grid_cube(assembly_cube_model, model_params_dim, 0.1, a[1:dim], b[1:dim], test_dists[j], 1000, Kmax)
            #print("\n", minim)
            errorsI[dim-1, j], errorsII[dim-1, j] = success_rate_assembly_cube(a, b, model_params_dim, MatToVec(minim, Maxim))
        end
    end

    #plot(test_dists, [errorsI[1, :], errorsI[2, :], errorsI[3, :], errorsI[4, :], errorsI[5, :]], ylim=[0, 0.002], grid=false, w=[1 1], linestyle=[:solid :solid], label=["eI (d=2)" "eI (d=3)" "eI (d=4)" "eI (d=5)" "eI (d=6)"])
    plot(test_dists, [errorsI[1, :], errorsI[2, :], errorsI[3, :], errorsI[4, :], errorsI[5, :], errorsI[6, :], errorsI[7, :], errorsI[8, :], errorsI[9, :]], ylim=[0, 0.002], grid=false, w=[1 1], linestyle=[:solid :solid], label=["eI (d=2)" "eI (d=3)" "eI (d=4)" "eI (d=5)" "eI (d=6)" "eI (d=7)" "eI (d=8)" "eI (d=9)" "eI (d=10)"])
    #plot(test_dists, [errorsII[1, :], errorsII[2, :], errorsII[3, :], errorsII[4, :], errorsII[5, :]], ylim=[0, 1], grid=false, w=[1 1], linestyle=[:solid], label=["eII (d=2)" "eII (d=3)" "eII (d=4)" "eII (d=5)" "eII (d=6)"])
    #=
    model_params_dim = [model_params[1][1:6], model_params[2][1:6]]
    adapt_X_cube, minim, Maxim = adaptive_grid_cube(assembly_cube_model, model_params_dim, 0.1, a[1:6], b[1:6], test_dists[10], 1000, Kmax)
    print("\n minim : ", minim)
    print("\n maxim : ", Maxim)
    print("\n retrieved : ", MatToVec(minim, Maxim))
    #showGrid(assembly_cube_model, model_params, 0.5, adapt_X_cube)
    =#
end
=#

# Cube models Test
begin
    # Parameters
    dim = 3
    model_params = [[[0.0, 1.0], [0.1, 0.6], [0, 0.9]], [[0.3, 0.7], [0.0, 1.0], [0.4, 1.0]]]
    #model_params= [[[0.0, 1.0], [0.0, 0.8]], [[-1.0, -0.4], [-0.9, -0.2]]]
    n_test = 40
    n0 = 10 # warning : no odd integer
    interpolation_inf = 0.9
    a = fill(-1.0, dim) # minimum boundaries
    b = fill(1.0, dim) # maximum boundaries
    K = 3 # maximal number of hypercubes to find (should be strictly greater than necessary)
    Ham_nums = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Change parameters depending on dimension
    model_params = [model_params[1][1:dim], model_params[2][1:dim]]

    # Grid Initialization
    test_X = regular_grid(a, b, n_test)
    print("test_X (regular) : success \n")
    adapt_X, minimlin, Maximlin = adaptive_grid_lin(assembly_cube_model, model_params, 0.1, a, b, n0)
    print("adapt_X (cutline) : success \n")
    adapt_X_cube, minim, Maxim, areas, _ = adaptive_grid_cube(assembly_cube_model, model_params, 0.1, a, b, 1e-1, 402, K, returns_full = true)
    print("adapt_X_cube (exploit) : success \n")
    quad_X = quad_fill(a, b, 2^dim)
    print("quad_X (regular) : success \n")
    Ham_X = Hammersley_grid(a, b, Ham_nums[1:(dim-1)], 1000)
    print("Ham_X (hammersley) : success \n")
    center_X = center_based_grid(a, b, 1000, 7)
    print("center_X (RCC) : success \n")
    # Test of Validity with a toymodel of linear metric interpolation
    #=
    N_test = n_test^dim
    alpha = interpolation_lin(assembly_cube_model, model_params, adapt_X, 1)
    val = [f_metric_lin(test_X[i, :], [adapt_X, 1, alpha]) for i=1:N_test]

    err = success_rate(assembly_cube_model, model_params, interpolation_inf, test_X, val)
    print("Taux erreur type I : ", err[1], "\n")
    print("Taux erreur type II : ", err[2])

    =#
    
    # metric interpolation test on Hammersley grid
    alpha_m = interpolation_lin(assembly_cube_model, model_params, Ham_X, 1)
    print("metric interpolation : success \n")
    val = [f_metric_lin(test_X[i, :], [alpha_m, Ham_X, 1]) for i=1:n_test^dim] # can take a lot of time, but necessary
    print("value compute : success \n")
    err = success_rate(assembly_cube_model, model_params, interpolation_inf, test_X, val)
    print("\n Taux erreur type I : ", err[1], "\n")
    print("Taux erreur type II : ", err[2])

    alpha_mcenter = interpolation_lin(assembly_cube_model, model_params, center_X, 1)
    val = [f_metric_lin(test_X[i, :], [alpha_mcenter, center_X, 1]) for i=1:n_test^dim]
    err = success_rate(assembly_cube_model, model_params, interpolation_inf, test_X, val)
    print("\n Taux erreur type I center : ", err[1], "\n")
    print("Taux erreur type II center : ", err[2])

    # gaussian interpolation
    sigmaused = maximum(b-a)/((1000)^(1/dim))
    alpha_g = interpolation_gauss(assembly_cube_model, model_params, Ham_X, sigmaused)

    # Show the results
    #showGrid(f_metric_lin, [alpha_m, Ham_X, 1], interpolation_inf, test_X)
    #showGrid(f_gauss, [alpha_g, Ham_X, sigmaused], interpolation_inf, test_X)
    #showGrid(assembly_cube_model, model_params, interpolation_inf, test_X)
    #showGrid(assembly_cube_model, model_params, interpolation_inf, center_X)
    #showGrid(cube_model, model_params, interpolation_inf, quad_X)
    #showGrid(assembly_cube_model, model_params, interpolation_inf, adapt_X)

    #display_sol(-1.0, 1.0, f_metric_lin, [alpha_m, Ham_X, 1])
    #display_sol(-1.0, 1.0, f_metric_lin, [alpha_mcenter, center_X, 1])
    #display_sol(-1.0, 1.0, f_gauss, [alpha_g, Ham_X, sigmaused])

    #showGrid(assembly_cube_model, model_params, interpolation_inf, Ham_X)
    #print(minim, "\n", Maxim, "\n retrieved with ", size(adapt_X_cube, 1), " points with dimension ", dim, "\n")
    #showGrid(assembly_cube_model, model_params, interpolation_inf, adapt_X_cube)
    #print(minim)
    #tK = length(minim[:, 1])
    #showGrid(assembly_cube_model, [[[minim[i,j], Maxim[i,j]] for j=1:dim] for i=1:tK], interpolation_inf, test_X)
    #print(minimlin, "\n", Maximlin, "\n retrieved with ", size(adapt_X, 1), " points with dimension ", dim, "\n")

    #=
    # Connex test
    CR_grid, CR_indexs, labels, areas = AbstractToConnected(retrieve_metric_model, [alpha_m, Ham_X, 1, interpolation_inf], interpolation_inf, a, b, 50)
    print("\n areas : ", areas)
    ind = 1
    cubes, areas = ConnectedToHypercube(4, CR_indexs[ind], labels, 50, dim)
    print("\n Kareas : ", areas)
    i = 4
    retrieve_params = [sort([CR_grid[cubes[i, 1], 1], CR_grid[cubes[i, 2], 1]]), sort([CR_grid[cubes[i, 1], 2], CR_grid[cubes[i, 2], 2]])]
    print("\n params 1 : ", retrieve_params)
    #showGrid(cube_model, retrieve_params, interpolation_inf, test_X)
    showGrid(assembly_cube_model, model_params, interpolation_inf, CR_grid[CR_indexs[ind], :])
    =#
    #=
    print(cubes)
    maxcubeall = cube_hull(CR_grid[CR_indexs[ind], :])
    print("maxhypercubegenelarized : ", maxcubeall)
    showGrid(cube_model, maxcubeall, interpolation_inf, test_X)
    max_assembly = assembly_cube_hull([CR_grid[CR_indexs[m], :] for m=1:length(CR_indexs)])
    #print("\n grid : ", cube_hull(CR_grid[CR_indexs[1], :]))
    showGrid(assembly_cube_model, max_assembly, interpolation_inf, test_X)
    =#
    #=
    # grid test
    alpha = interpolation_lin(assembly_cube_model, model_params, center_X, 1)
    val = [f_metric_lin(test_X[i, :], [center_X, 1, alpha]) for i=1:(n_test^dim)]
    err = success_rate(assembly_cube_model, model_params, interpolation_inf, test_X, val)
    print("\n Taux erreur type I : ", err[1], "\n")
    print("Taux erreur type II : ", err[2])
    display_sol_lin(-1.0, 1.0, alpha, center_X, 1)
    =#
    # separation test
    Id1, Id0 = works(assembly_cube_model, model_params, 0.5, Ham_X)
    params, errIIsep1, errIIsep2 = SeparateAndCut(assembly_cube_model, model_params, Ham_X, Id1, Id0, 0.0)
    print("\n error type II algorithm evaluation : ", 1/(1+min(errIIsep1, errIIsep2)))
    for i=1:length(params)
        print("\n params", i, " = ", params[i], " : area = ", cube_measurement(params[i]))
    end
    errorI, errorII = success_rate_assembly_cube(a, b, model_params, params)
    print("\n error type I : ", errorI, " & error type II : ", errorII)
    #showGrid(assembly_cube_model, params, interpolation_inf, test_X)
end
