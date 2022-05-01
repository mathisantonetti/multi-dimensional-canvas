using Pkg
using Plots
using LinearAlgebra

### Tools
# Logic Tools
# tests if e (vector) is eps-close to one element (vector) in L
function oneclose(e, L, eps)
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

# returns true if the hypercube H1 is (strictly) included in one element H2 in list_H2, H1 & H2 given with their parameters [[inf_dim1, inf_dim2, ...], [sup_dim1, ...]]
function isIncludedInOneHC(H1, list_H2)
    d = size(H1, 2)
    n = size(list_H2, 1)
    for i=1:n
        count = 0
        for j=1:d
            if(H1[1, j] <= (list_H2[i])[1, j] || (list_H2[i])[2, j] <= H1[2, j]) # if we can have x in H1 and not in the interior of list_H2[i]
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

# returns the decomposition in the base of n with strict maximal n of base^d 
function decompo_base(n, d; base=10)
    res = zeros(Int, d)
    x0 = n
    for i=1:d
        q = floor(Int, x0/base^(d-i))
        res[i] = q
        x0 = x0 - q*base^(d-i)
    end
    return res
end

# Grid tools
# returns a regular grid in dimension dim with boundaries [a[1], b[1]]x...x[a[dim], b[dim]]
function regular_grid(a, b, N)
    dim = length(a)
    Y = zeros((N^dim, dim))
    c = LinRange(a[dim] + (b[dim]-a[dim])/(2N), b[dim] - (b[dim]-a[dim])/(2N), N)
    if(dim >= 2)
        Z = regular_grid(a[1:(dim - 1)], b[1:(dim-1)], N)
        for i=1:N^(dim - 1)
            for j=1:N
                Y[(i-1)*N+j, :] = [Z[i], c[j]]
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

# one step for the center based grid sequence, receive Vn and returns Vn+1
function step_seq_center(Vn, s)
    kn = Vn[1]
    jn = Vn[2]
    if(jn >= 3)
        return [kn+1, jn-2]
    else
        return [max(2kn+jn-s + 1, 1), s-1-abs(2kn+jn-s)]
    end
end

function center_line(a, b, V)
    k = V[1]
    j = V[2]
    d = length(a)
    N = d*(2^k - 1)^d*(2^(k + j - 1) - 1)
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
    current_V = [1, 1]
    j = 1
    X = zeros(Float64, 0)
    sc = ceil(Int, s)
    while(j <= N && current_V[2] >= 1)
        Line, L = center_line(a, b, current_V)
        #print("\n current line : ", Line)
        for i=1:L
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
        m = c[n0 - count] + (b[i]-a[i])/(2n0)
        countmax = count
    end

    return Y, m, M
end

# fills the holes with 2^dim point in the hypercube generated by [[m[1], M[1]], ..., [m[d], M[d]]
function quad_fill(dim, m, M)
    Y = zeros((2^dim, dim))
    if(dim >= 2)
        Z = quad_fill(dim - 1, m, M)
        for i=1:2^(dim - 1)
            Y[2i-1, :] = [Z[i], m[dim] + 3*(M[dim] - m[dim])/4]
            Y[2i, :] = [Z[i], m[dim] + (M[dim] - m[dim])/4]
        end
        return Y
    else
        Y[1, :] = [m[1] + 3*(M[1] - m[1])/4]
        Y[2, :] = [m[1] + (M[1] - m[1])/4]
        return Y
    end
end

# returns an adaptive grid generated with cutlines and quadfills with boundaries [a[1], b[1]]x...x[a[dim], b[dim]]
function adaptive_grid_lin(model, params, ad_inf, a, b, n0, dim)
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
    X[(dim*n0+1):(dim*n0+2^dim), :] = quad_fill(dim, m, M)
    return X
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
function adaptive_grid_cube(model, params, ad_inf, a, b, nominal_dist, Nmax, K)
    # Initialization
    dim = length(a)
    m = ones((K, dim))
    M = ones((K, dim))
    Kareas = fill(-7.0, K)
    X_ada = zeros((0, dim))

    # Base of lodes
    X_base = zeros((2^dim+1, dim))
    X_base[1, :] = (a + b)/2.0
    X_base[2:2^dim+1, :] = quad_fill(dim, a, b)
    for i=2:min(floor(Int, (Nmax-1)/(2^dim)), 2^dim+1)
        X_base = cat(X_base, quad_fill(dim, min.(2*X_base[i, :], 0), max.(2*X_base[i, :], 0)), dims=1)
    end

    L_base = size(X_base, 1)

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
                    return X_ada
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
                                return X_ada
                            end
                        end
                        if(model(candidates_X[k, :], params) <= ad_inf)
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
                                return X_ada
                            end
                        end
                        if(model(candidates_X[k, :], params) <= ad_inf)
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
                print(area, " : ", Kareas, "\n")
                for k=1:(K-1)
                    if(Kareas[k] < area && area < Kareas[k+1])
                        Mcat = cat(M[2:k, :], reshape(maxi, 1, dim), dims=1) # add the element
                        M = cat(Mcat, M[(k+1):K, :], dims=1)
                        mcat = cat(m[2:k, :], reshape(mini, 1, dim), dims=1)
                        m = cat(mcat, m[(k+1):K, :], dims=1)
                        Kareascat = cat(Kareas[2:k], [area], dims=1)
                        Kareas = cat(Kareascat, Kareas[(k+1):K], dims=1)
                        break
                    end
                end
                if(Kareas[K] < area)
                    M[1:(K-1), :] = copy(M[2:K, :])
                    m[1:(K-1), :] = copy(m[2:K, :])
                    Kareas[1:(K-1)] = copy(Kareas[2:K])
                    M[K, :] = maxi
                    m[K, :] = mini
                    Kareas[K] = area
                end
            end
        end
    end

    return X_ada, m, M
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

# distance matrix
function M(X, p)
    n = size(X, 1)
    Mat = zeros(n, n)
    for i=1:n
        for j=1:n
            Mat[i,j] = dist(X[i, :], X[j, :])^p
        end
    end
    return Mat
end

# metric linear interpolation
function interpolation_lin(model, params, X, p)
    N = size(X, 1)
    beta = [model(X[i, :], params) for i=1:N]
    return M(X, p)\beta
end

# returns f(x) = sum_i alpha_i d(x, X_i)^p with params = [X, p, alpha]
function f_metric_lin(x, params)
    X = params[1]
    p = params[2]
    alpha = params[3]
    Ntrain = size(X, 1)
    return sum([alpha[i]*dist(x, X[i, :])^p for i=1:Ntrain])
end

# Converting Tools

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
                n = length(sorted_labels)

                # fuse all regions in the neighbourhood
                for k=2:(n-1)
                    slabels_k = sorted_labels[k]
                    slabels_kp = sorted_labels[k+1]
                    append!(CRindexs[slabels_k], CRindexs[slabels_kp]) # add region at i+1 to i
                    areas[slabels_k] += areas[slabels_kp] # add volume
                    for ind in CRindexs[slabels_kp]
                        CRlabel_test_X[ind] = slabels_k
                    end
                    deleteat!(CRindexs, slabels_kp) # remove region at i+1
                    deleteat!(areas, slabels_kp)
                end
                
                # add the element to the region
                CRlabel_test_X[i] = sorted_labels[2]
                
                min_previous_n = minimum(nlabels[(d+1):(2d)])
                # add the element if it is on the region boundary
                if(min_previous_n == 0)
                    append!(CRindexs[sorted_labels[2]], i)
                end

                # volume calculus
                areas[sorted_labels[2]] += compute_volume(i, CRlabel_test_X, N_test, d)
            
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

    return test_X, CRindexs, areas
end

#= 
---
parameters :
_ K : number max of hypercubes 
_ boundary_inds : boundary points indexs (sorted) from a regular grid N_test^dim
_ N_test : accuracy of the regular grid (number of points per line)
_ dim : dimension of the elements of the grid
---
returns the K maximal hypercubes found in the connected region
=#

function ConnectedToHypercube(K, boundary_inds, N_test, dim)
    N = length(boundary_inds)
    digit_BI = [decompo_base(boundary_inds[i] - 1, dim, base=N_test) for i=1:N]
    Kareas = fill(-1.0, K)
    Kcubes_inds = zeros(Int, (K, 2)) 
    for i=1:N
        # To Do : recursive
        for j=(i+1):N
            area = exp(sum([log(abs(digit_BI[i][k] - digit_BI[j][k]) + (digit_BI[i][k] == digit_BI[j][k])*1e-8) for k=1:dim]))
            stopvar = false
            for l=1:N
                count = 0
                for k=1:dim
                    if((digit_BI[i][k] < digit_BI[l][k] && digit_BI[l][k] < digit_BI[j][k]) || (digit_BI[i][k] > digit_BI[l][k] && digit_BI[l][k] > digit_BI[j][k]))
                        count += 1
                    end
                end
                if(count == dim)
                    stopvar = true
                    break
                end
            end
            if(stopvar)
                break
            end
            for k=1:K
                if(k == K)
                    if(Kareas[k] < area)
                        if(k >= 2)
                            Kareas[1:(k-1)] = copy(Kareas[2:k])
                            Kcubes_inds[1:(k-1), :] = copy(Kcubes_inds[2:k, :])
                        end
                        Kareas[k] = area
                        Kcubes_inds[k, 1] = boundary_inds[i]
                        Kcubes_inds[k, 2] = boundary_inds[j]
                    end
                elseif(Kareas[k] < area && area <= Kareas[k+1])
                    if(k >= 2)
                        Kareas[1:(k-1)] = copy(Kareas[2:k])
                        Kcubes_inds[1:(k-1), :] = copy(Kcubes_inds[2:k, :])
                    end
                    Kareas[k] = area
                    Kcubes_inds[k, 1] = boundary_inds[i]
                    Kcubes_inds[k, 2] = boundary_inds[j]
                    break
                end
            end
        end
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

# displays the metric linear interpolation solution based on the pre-computed solution alpha and the grid X
function display_sol_lin(a, b, alpha, X, p)
    N = length(alpha)
    x = LinRange(a, b, 100)
    y = LinRange(a, b, 100)
    col = zeros(100, 100)
    mini = 1.0
    maxi = 0.0
    for i=1:100
        for j=1:100
            val = sum([alpha[k]*dist([x[i], y[j]], X[k, :])^p for k=1:N])
            #col[j,i] = val + (1 - val)*(val > 1) - val*(val < 0)
            col[j,i] = (1/(1+exp(-val)))
            if(col[j, i] < mini)
                mini = col[j, i]
            end
            if(col[j, i] > maxi)
                maxi = col[j, i]
            end
        end
    end
    col = (col .- mini)/(maxi - mini)
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
        if(model(grid[i, :], params) > eval_inf)
            count_reals += 1
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

# retrieve the model from metric interpolation with params = [alpha, grid, inter_inf] (where alpha is the interpolation restult, grid is the grid used to interpolate, and inter_inf is the sensibility)
function retrieve_model(x, params)
    alpha = params[1]
    grid = params[2]
    interpolation_inf = params[3]
    N_train = size(grid, 1)
    return sum([alpha[k]*dist(x, grid[k, :])^1 for k=1:N_train]) > interpolation_inf
end

# Cube models Test
begin
    # Parameters
    dim = 2
    n_test = 70
    n0 = 10 # warning : no odd integer
    interpolation_inf = 0.8
    model_params = [[[0.0, 1.0], [0.0, 0.8]], [[-1.0, -0.4], [-0.9, -0.2]]]
    a = fill(-1.0, dim) # minimum boundaries
    b = fill(1.0, dim) # maximum boundaries
    K = 5 # maximal number of hypercubes to find (should be strictly greater than necessary)

    # Grid Initialization
    test_X = regular_grid(a, b, n_test)
    adapt_X = adaptive_grid_lin(assembly_cube_model, model_params, 0.1, a, b, n0, 2)
    adapt_X_cube, minim, Maxim = adaptive_grid_cube(assembly_cube_model, model_params, 0.1, a, b, 3e-1, 1000, K)
    quad_X = quad_fill(dim, a, b)
    Ham_X = Hammersley_grid(a, b, [2], 24)
    center_X = center_based_grid(a, b, 24, 4)

    # Test of Validity with a toymodel of linear metric interpolation
    #=
    N_test = n_test^dim
    alpha = interpolation_lin(assembly_cube_model, model_params, adapt_X, 1)
    val = [f_metric_lin(test_X[i, :], [adapt_X, 1, alpha]) for i=1:N_test]

    err = success_rate(assembly_cube_model, model_params, interpolation_inf, test_X, val)
    print("Taux erreur type I : ", err[1], "\n")
    print("Taux erreur type II : ", err[2])

    =#

    # interpolation test
    alpha = interpolation_lin(assembly_cube_model, model_params, Ham_X, 1)
    val = [f_metric_lin(test_X[i, :], [Ham_X, 1, alpha]) for i=1:n_test^dim]
    err = success_rate(assembly_cube_model, model_params, interpolation_inf, test_X, val)
    print("\n Taux erreur type I : ", err[1], "\n")
    print("Taux erreur type II : ", err[2])


    # Show the results
    #showGrid(retrieve_model, [alpha, center_X, interpolation_inf], interpolation_inf, test_X)
    showGrid(assembly_cube_model, model_params, interpolation_inf, test_X)
    #showGrid(assembly_cube_model, model_params, interpolation_inf, center_X)
    #showGrid(cube_model, model_params, interpolation_inf, quad_X)
    #showGrid(cube_model, model_params, interpolation_inf, adapt_X)

    #display_sol_lin(-1.0, 1.0, alpha, adapt_X, 1)
    #display_sol_lin(-1.0, 1.0, alpha, Ham_X, 1)
    #display_sol_lin(-1.0, 1.0, alpha, center_X, 1)

    #showGrid(assembly_cube_model, model_params, interpolation_inf, Ham_X)
    #print(minim, "\n", Maxim, "\n retrieved with ", size(adapt_X_cube, 1), " points with dimension ", dim, "\n")
    #showGrid(assembly_cube_model, model_params, interpolation_inf, adapt_X_cube)
    #showGrid(assembly_cube_model, [[[minim[i,j], Maxim[i,j]] for j=1:dim] for i=1:K], interpolation_inf, test_X)
    #=
    # Connex test
    CR_grid, CR_indexs, areas = AbstractToConnected(retrieve_model, [alpha, Ham_X, interpolation_inf], interpolation_inf, a, b, 20)
    print("\n areas : ", areas)
    ind = 1
    cubes, areas = ConnectedToHypercube(4, CR_indexs[ind], 20, dim)
    print("\n Kareas : ", areas)

    showGrid(assembly_cube_model, model_params, interpolation_inf, CR_grid[CR_indexs[ind], :])
    print(cubes)
    i = 1
    retrieve_params = [sort([CR_grid[cubes[i, 1], 1], CR_grid[cubes[i, 2], 1]]), sort([CR_grid[cubes[i, 1], 2], CR_grid[cubes[i, 2], 2]])]
    print("\n params 1 : ", retrieve_params)
    showGrid(cube_model, retrieve_params, interpolation_inf, test_X)
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
end