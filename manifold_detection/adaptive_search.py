import tools.manifolds
import matplotlib.pyplot as plt

torus = tools.manifolds.Torus([0.0, 0.0], 1.0, 3.0)

model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [0.6, 1.0]), tools.manifolds.Cube([0.0, 0.7], [0.3, 1.0])])

Id1, Id0 = works(model, eval_inf, X)
params, errIIsep1, errIIsep2 = SeparateAndCut(grid, Id1, Id0, 0.0)

print(params)