using Pkg
using Plots
using LinearAlgebra

### Tools
# Grid tools
# returns a regular grid in dimension dim duplicating the line c in every dimension
function line_fill(dim, c)
    N = length(c)
    Y = zeros((N^dim, dim))
    if(dim >= 2)
        Z = line_fill(dim - 1, c)
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

# returns a regular grid in dimension dim
function regular_grid(a, b, N, dim)
    c = LinRange(a+(b-a)/(2N), b-(b-a)/(2N), N)
    return line_fill(dim, c)
end

# returns the cut (deduced from the model with condition model > ad_inf) of the line going through the dimension i
function cutline(model, params, ad_inf, line, a, b, n0, i)
    dim = length(line) + 1
    Y = zeros((n0, dim))
    c = LinRange(a+(b-a)/(2n0), b-(b-a)/(2n0), n0)
    count, countmax, m, M = 0, 0, a, b
    for j = 1:n0
        Y[j, 1:(i-1)] = line[1:(i-1)]
        Y[j, i] = c[j]
        Y[j, (i+1):dim] = line[i:(dim-1)]
        if(model(Y[j, :], params) <= ad_inf) # majority election
            if(count > countmax)
                M = c[j] - (b-a)/(2n0)
                m = c[j - count] - (b-a)/(2n0)
                countmax = count
            end
            count = 0
        else
            count += 1
        end
    end
    return Y, m, M
end

# fills the holes with one point
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

# returns an adaptive grid generated with cutlines and quadfills
function adaptive_grid_lin(model, params, ad_inf, a, b, nbpoints, n0, dim)
    middle = fill((a + b)/2, dim - 1)
    X = zeros((n0*dim + 2^dim, dim))
    m = fill(a, dim)
    M = fill(b, dim)
    for d=1:dim
        X[(n0*(d-1)+1):(n0*d), :], m[d], M[d] = cutline(model, params, ad_inf, middle, a, b, n0, d)
        if(d <= dim - 1)
            middle[d] = (m[d] + M[d])/2
        end
    end
    X[(dim*n0+1):(dim*n0+2^dim), :] = quad_fill(dim, m, M)
    return X
end

# Toymodels

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

# Calculus tools

# logarithm adapted to eps (because it converges toward -infinity when x converges toward 0)
function invexp(x, eps)
    return log(eps)*(0 <= x <= eps) + log(x)*(x > eps)
end

# metric distance
function d(x, y)
    return norm(x - y)
end

# distance matrix
function M(X, p)
    n = size(X, 1)
    Mat = zeros(n, n)
    for i=1:n
        for j=1:n
            Mat[i,j] = d(X[i, :], X[j, :])^p
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

# metric pseudo-gaussian interpolation (almost the same)
function interpolation_exp(X, p, model, eps)
    N = size(X, 1)
    beta = [invexp(model(X[i, :]), eps) for i=1:N]
    return M(X, p)\beta
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
    Id0 = zeros(Int64, 0) # points working
    Id1 = zeros(Int64, 0) # points not working
    for i=1:N
        if(model(X[i, :], params) > eval_inf)
            append!(Id0, i)
        else
            append!(Id1, i)
        end
    end
    return Id0, Id1
end

# Canvas Tools

# shows the grid X with the separation of good and bad points for the model
function showGrid(model, params, eval_inf, X)
    Idex0, Idex1 = works(model, params, eval_inf, X)
    tuple_real_X0 = ListToTuple([X[i, :] for i in Idex0])
    tuple_real_X1 = ListToTuple([X[i, :] for i in Idex1])
    scatter(tuple_real_X0, xlims = (-1, 1), ylims = (-1, 1), color = "blue", label = "success")
    scatter!(tuple_real_X1, xlims = (-1, 1), ylims = (-1, 1), color = "red", label = "fail")
end

# displays the metric linear interpolation solution based on the pre-computed solution alpha and the grid X
function display_sol_lin(a, b, alpha, X, p)
    N = length(alpha)
    x = LinRange(a, b, 100)
    y = LinRange(a, b, 100)
    col = zeros(100, 100)
    for i=1:100
        for j=1:100
            val = sum([alpha[k]*d([x[i], y[j]], X[k, :])^p for k=1:N])
            col[j,i] = val + (1 - val)*(val > 1)
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

# retrieve the model with params = [alpha, grid, inter_inf] (where alpha is the interpolation restult, grid is the grid used to interpolate, and inter_inf is the sensibility)
function retrieve_model(x, params)
    alpha = params[1]
    grid = params[2]
    interpolation_inf = params[3]
    N_train = size(grid, 1)
    return sum([alpha[k]*d(x, grid[k, :])^1 for k=1:N_train]) > interpolation_inf
end

#=
# Ball models Test
begin
    # Parameters
    n_test = 70
    n0 = 10 # warning : no odd integer
    eps = 0.1
    interpolation_inf = 0.86
    model_params = [[0.2, 0.0], 0.4, 0.8]

    # Grid Initialization
    test_X = regular_grid(-1.0, 1.0, n_test, 2)
    real_X = regular_grid(-1.0, 1.0, n0, 2)
    adapt_X = adaptive_grid_lin(ball_model, model_params, 0.1, -1.0, 1.0, 3, n0, 2)

    # Test of Validity with a toymodel of linear metric interpolation
    N_test = n_test^2
    N_train = n0*2 + 2^2
    alpha = interpolation_lin(ball_model, model_params, adapt_X, 1)
    val = [sum([alpha[k]*d(test_X[i, :], adapt_X[k, :])^1 for k=1:N_train]) for i=1:(n_test^2)]

    err = success_rate(ball_model, model_params, interpolation_inf, test_X, val)
    print("Taux erreur type I : ", err[1], "\n")
    print("Taux erreur type II : ", err[2])

    # Show the results
    #showGrid(ball_model, model_params, test_X)
    #showGrid(real_X)
    #showGrid(ball_model, model_params, adapt_X)

    #display_sol_lin(-1.0, 1.0, alpha, adapt_X, 1)
end
=#

# Cube models Test
begin
    # Parameters
    n_test = 70
    n0 = 10 # warning : no odd integer
    interpolation_inf = 0.9749
    model_params = [[-0.4, 1.0], [-1.0, 0.7]]

    # Grid Initialization
    test_X = regular_grid(-1.0, 1.0, n_test, 2)
    real_X = regular_grid(-1.0, 1.0, n0, 2)
    adapt_X = adaptive_grid_lin(cube_model, model_params, 0.1, -1.0, 1.0, 3, n0, 2)

    # Test of Validity with a toymodel of linear metric interpolation
    N_test = n_test^2
    N_train = n0*2 + 2^2
    alpha = interpolation_lin(cube_model, model_params, adapt_X, 1)
    val = [sum([alpha[k]*d(test_X[i, :], adapt_X[k, :])^1 for k=1:N_train]) for i=1:(n_test^2)]

    err = success_rate(cube_model, model_params, interpolation_inf, test_X, val)
    print("Taux erreur type I : ", err[1], "\n")
    print("Taux erreur type II : ", err[2])

    # Show the results
    showGrid(retrieve_model, [alpha, adapt_X, interpolation_inf], interpolation_inf, test_X)
    #showGrid(cube_model, model_params, interpolation_inf, test_X)
    #showGrid(real_X)
    #showGrid(cube_model, model_params, interpolation_inf, adapt_X)

    #display_sol_lin(-1.0, 1.0, alpha, adapt_X, 1)
end