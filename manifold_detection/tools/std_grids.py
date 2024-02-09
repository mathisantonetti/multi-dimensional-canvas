import numpy as np
import matplotlib.pyplot as plt
import manifold_detection.tools.manifolds

class Grid():
    def __init__(self, dim, N):
        '''
        Parameters :
        _ dim : dimension of the points
        _ N : number of points in the grid
        '''
        self.N = N
        self.dim = dim
        self.points = np.zeros((N, self.dim))

    def add(self, points):
        self.N += points.shape[0]
        self.points = np.append(self.points, points, axis=0)

    def works(self, model, eval_inf):
        Id1 = [] # points working
        Id0 = [] # points not working
        for i in range(self.N):
            if(model(self.points[i, :]) > eval_inf):
                Id1.append(i)
            else:
                Id0.append(i)

        return Id1, Id0

    def show(self, model, eval_inf):
        if(self.dim == 2):
            Id1, Id0 = self.works(model, eval_inf)
            plt.scatter(self.points[Id1,0], self.points[Id1,1], color="blue")
            plt.scatter(self.points[Id0, 0], self.points[Id0, 1], color="red")
            plt.show()

    def median(self, indices=None, return_indices=False, plans=None):
        '''
        Parameters :
        _ indices : list of indices of the points to take into account (default : None)
        _ return_indices : if True, returns the indices of the points in the optimal separations (default : False)
        _ plans : hyperplans giving the level surfaces for the order used to calculate the median (default : hypercube faces).

        Returns :
        _ ind : index of the median point
        _ list_inds : list of the indices of the points in the separations made with the median point (if return_indices = True)
        '''

        if(indices is None):
            indices = [i for i in range(self.N)]

        if(plans is None):
            plans = np.eye(self.dim)

        n = len(indices)

        Nplan = plans.shape[0]

        ind = 0
        list_inds = [[] for i in range(2*Nplan)]
        opt_value = np.inf
        for i in indices:
            x = self.points[i]
            new_list_inds = [[] for k in range(2*Nplan)]
            for j in indices:
                if(j == i):
                    continue # skip the point i

                y = self.points[j]
                for d in range(Nplan):
                    if(np.dot(plans[d], y - x) > 0.0):
                        new_list_inds[2*d].append(j)
                    else:
                        new_list_inds[2*d+1].append(j)
            p = [len(new_list_inds[2*d]) for d in range(Nplan)]
            new_value = np.sum([np.abs(p[k] - (n-1)/2) for k in range(Nplan)])
            if(new_value < opt_value):
                list_inds = new_list_inds
                ind = i
                opt_value = new_value

        if(return_indices):
            return ind, list_inds
        else:
            return ind

    def extension(self, a, b, manifold_type="Cube"):
        for point in self.points:
            point_extension = Cube(a, b, index=point, grid=self)



class RegularGrid(Grid):
    def __init__(self, a, b, N):
        '''
        Parameters :
        _ a : minimum boundaries of [a, b]
        _ b : maximum boundaries of [a, b]
        _ N : number of points per coordinates
        '''
        dim = len(a)
        super().__init__(dim, N**dim)
        c = np.linspace(a[self.dim-1] + (b[self.dim-1]-a[self.dim-1])/(2*N), b[self.dim-1] - (b[self.dim-1]-a[self.dim-1])/(2*N), N)
        if(self.dim >= 2):
            subgrid = RegularGrid(a[:(self.dim - 1)], b[:(self.dim-1)], N)
            for i in range(N**(self.dim - 1)):
                for j in range(N):
                    self.points[(i-1)*N+j, :(self.dim - 1)] = subgrid.points[i, :(self.dim - 1)]
                    self.points[(i-1)*N+j, self.dim-1] = c[j]
        else:
            for j in range(N):
                self.points[j, :] = [c[j]]

class CenteredGrid(Grid):
    def __init__(self, a, b, N, s, model=None, inv_detection=False):
        '''
        Parameters :
        _ a : minimum boundaries of [a, b]
        _ b : maximum boundaries of [a, b]
        _ N : maximal number of points in the grid
        _ s : control of the grid sequence (represents the minimum distance between points of the grid)
        _ model : model to stop the grid generation. If model=None, the interruption is ignored  (default : None)
        _ inv_detection : If True (resp. False), stop the grid generation when the model is negative (resp. positive) for a point.
        '''
        a = np.array(a)
        b = np.array(b)
        self.dim = a.shape[0]
        self.points = np.zeros((0,self.dim))
        self.evals = []
        current_V = [1, 1]
        j = 1
        X = np.zeros(0)
        sc = int(s)+(np.abs(s - int(s)) > 10**(-10))
        while(j <= N and current_V[1] >= 1):
            Line, L = self.center_line(a, b, current_V)
            for i in range(L):
                self.points = np.concatenate((self.points, np.reshape(Line[i, :], (1,self.dim))), axis=0) # add the element
                j += 1
                if(j > N):
                    break
                if(not(model is None)):
                    test = model(Line[i,:])
                    self.evals.append(test)
                    if(test < 0.5 and inv_detection):
                        break
                    elif(test > 0.5 and (inv_detection == False)):
                        break

            if(self.evals[-1] < 0.5 and inv_detection):
                break
            elif(self.evals[-1] > 0.5 and (inv_detection == False)):
                break

            current_V = self.step_seq_center(current_V, sc)

        self.N = j-1

    def step_seq_center(self, Vn, s):
        """
        Parameters :
        _ Vn : n-th term of the sequence Vs
        _ s : control of the sequence Vs

        Returns :
        _ V(n+1) : (n+1)-th term of the sequence Vs
        """
        kn = Vn[0]
        jn = Vn[1]
        if(jn >= 3):
            return [kn+1, jn-2]
        else:
            return [max(2*kn+jn-s + 1, 1), s-1-abs(2*kn+jn-s)]

    def center_line(self, a, b, V):
        '''
        Parameters :
        _ a : minimum boundaries of [a, b]
        _ b : maximum boundaries of [a, b]
        _ V : tuple (k, j) giving the decision for drawing the lines

        Returns :
        _ Matrix Line which is the grid corresponding to V
        _ int i - 1 : length of Line
        '''
        k = V[0]
        j = V[1]
        d = a.shape[0]
        N = d*(2**((k-1)*d))*(2**(j - 1))
        Line = np.zeros((N, d))
        i = 0
        for t in range(2**(d*(k-1))):
            n = []
            r = 0
            t0 = t
            for l in range(d):
                r = t0%(2**(k-1))
                t0 = int((t0 - r)/(2**(k-1)))
                n.append(2*r+1)
            if(j == 1):
                Line[i, :] = (np.array(n)/(2**k))*(b - a) + a
                i += 1
            else:
                for p in range(2**(j - 2)):
                    for l in range(d):
                        Line[i, :] = (np.array(n)/(2**k))*(b - a) + a
                        Line[i+1, :] = Line[i, :]
                        Line[i, l] = Line[i, l] + ((2*p+1)/(2**(k + j - 1)))*(b[l] - a[l])
                        Line[i+1, l] = Line[i+1, l] - ((2*p+1)/(2**(k + j - 1)))*(b[l] - a[l])
                        i += 2
        return Line[:i, :], i

class LowDiscrepancyGrid(Grid):
    def __init__(self,a,b,c,N):
        '''
        Parameters :
        _ a : minimum boundaries of [a, b]
        _ b : maximum boundaries of [a, b]
        _ c : base for Van Der Corput function : [c_1, ..., c_(d-1)] where pgcd(c_i, c_j) = 1 for all i not equal to j
        _ N : number of points
        '''
        s = len(c) # = d - 1
        super().__init__(s+1, N)
        for i in range(N):
            for j in range(s):
                self.points[i, j] = (b[j] - a[j])*self.Van_Der_Corput_g(c[j], i) + a[j]
            self.points[i, s] = (b[s] - a[s])*i/N + a[s]

    def Van_Der_Corput_g(self, c, n):
        '''
        Parameters :
        _ c : decoding base
        _ n : number to represent

        Returns :
        sum_i d_i c^(-i) where n = sum_i d_i c^(i-1)
        '''
        m = n
        dk = []
        while(m >= 2):
            r = m%c
            dk.append(r)
            m = int((m-r)/c)
        dk.append(1)
        L = len(dk)
        return np.sum([dk[i]*float(c)**(-(i+1)) for i in range(L)])