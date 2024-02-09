import numpy as np

class Atlas():
    def __init__(self, manifolds, parameters=None, shape=None, error_max=0.05, number_manifold_limit=100):
        self.manifolds = manifolds

        if((shape is None) == False):
            volmax = np.inf
            params = None
            if(shape == "Cube"):
                params = [Cube(parameters[i][0], parameters[i][1]) for i in range(len(parameters))]
            elif(shape == "Torus"):
                params = [Torus(parameters[i][0], parameters[i][1], parameters[i][2]) for i in range(len(parameters))]

            for k in range(number_manifold_limit):
                param_indices = np.argsort([params[i].measure() - (self.intersect(params[i])).measure() for i in range(len(params))])
                #print(sols.intersect(props[prop_indices[-1]]))
                if(k == 0):
                    volmax = params[param_indices[-1]].measure() - (self.intersect(params[param_indices[-1]])).measure()
                elif((params[param_indices[-1]].measure() - (self.intersect(params[param_indices[-1]])).measure())/volmax < error_max):
                    break
                self.update([params[param_indices[-1]]])

    def update(self, manifolds):
        self.manifolds = self.manifolds + manifolds

    def intersect(self, manifold):
        K = len(self.manifolds)
        intersect_atlas = Atlas([self.manifolds[k].intersect(manifold) for k in range(K)])
        return intersect_atlas

    def measure(self):
        #print(self.manifolds)
        K = len(self.manifolds)
        if(K == 0):
            return 0.0
        elif(K == 1):
            return self.manifolds[0].measure()
        else:
            for i in range(K):
                if(self.manifolds[i].measure() < 10**(-10)):
                    new_atlas = Atlas(self.manifolds[:i])
                    if(i < K-1):
                        new_atlas.update(self.manifolds[i+1:])
                    return new_atlas.measure()

            intersect_atlas = Atlas([manifold.intersect(self.manifolds[K-1]) for manifold in self.manifolds[:K-1]])
            big_atlas = Atlas(self.manifolds[:K-1])

            return big_atlas.measure() - intersect_atlas.measure() + self.manifolds[K-1].measure()

    def __call__(self, x):
        v = np.sum([manifold(x) for manifold in self.manifolds])
        return 2*v/(1+v)

    def __str__(self):
        s = ""
        for manifold in self.manifolds:
            s += manifold.__str__() + ", "

        return "[" + s[:-2] + "]"


class Cube():
    def __init__(self, a, b, index=None, grid=None):
        self.a = a
        self.b = b

        if(not(self.a is None) and not(self.b is None)):
            self.a = np.array(self.a)
            self.b = np.array(self.b)
            self.d = self.b.shape[0]

        if(not(index is None) and not(grid is None)):
            new_a, new_b = np.copy(a), np.copy(b)
            dims_to_explore = [i for i in range(2*grid.dim)]

            for dim in range(2*grid.dim):
                val = np.inf
                index_val = -1
                dim_to_remove = -1
                for i in range(len(grid.points)):
                    if(i == index):
                        continue

                    flag = False
                    for s in range(grid.dim):
                        if(grid.points[i][s] >= new_b[s] or grid.points[i][s] <= new_a[s]):
                            flag = True
                            break

                    if(flag):
                        continue

                    cand_val, cand_dim = -1, -1
                    for s in dims_to_explore:
                        new_val = ((-1)**s)*(grid.points[i][s//2] - grid.points[index][s//2])
                        #print(s, " , ", new_val, " : ", val)
                        if(new_val > 0):
                            if(new_val > cand_val):
                                cand_val = new_val
                                cand_dim = s

                    if(cand_val < val and cand_dim != -1):
                        val = cand_val
                        dim_to_remove = cand_dim
                        index_val = i

                if(dim_to_remove == -1):
                    break

                s = dim_to_remove
                if(s%2 == 0):
                    new_b[s//2] = grid.points[index_val][s//2]
                else:
                    new_a[s//2] = grid.points[index_val][s//2]
                dims_to_explore.remove(s)

            self.a = new_a
            self.b = new_b

    def measure(self):
        if(self.a is None or self.b is None):
            return 0.0
        else:
            return np.prod(self.b-self.a)

    def __call__(self, x):
        for i in range(self.d):
            if(self.a[i] > x[i] or x[i] > self.b[i]): # if x is not in the "cube"
                return 0.0 # False
        return 1.0 # True

    def intersect(self, other):
        new_a = []
        new_b = []
        for i in range(self.d):
            new_a.append(max(self.a[i], other.a[i]))
            new_b.append(min(self.b[i], other.b[i]))
            if(new_a[i] > new_b[i]):
                new_a = None
                new_b = None
                break
        intersection = Cube(new_a, new_b)
        return intersection

    def __str__(self):
        return "Cube(" + self.a.__str__() + ", " + self.b.__str__() + ")"

class Torus():
    def __init__(self, center, r_int, r_ext):
        self.r_int = r_int
        self.r_ext = r_ext
        self.center = np.array(center)

    def measure(self):
        return np.pi*(self.r_ext**2-self.r_int**2)

    def __call__(self, x):
        x = np.array(x)
        return (np.linalg.norm(x - self.center) < self.r_ext) +(np.linalg.norm(x-self.center) > self.r_int)
