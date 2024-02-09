import manifold_detection.tools as tools
from manifold_detection.aposteriori_search import *
from manifold_detection.adaptive_search import *
from scipy.spatial import Voronoi, voronoi_plot_2d

def median_test(i):
    if(i == 0):
        grid = tools.std_grids.CenteredGrid([0.0, 0.0], [1.0, 1.0],  10, 10)
        model = tools.manifolds.Cube([0.0, 0.0], [1.0, 1.0])
        grid.show(model, 0.5)

        index = grid.median(plans=np.array([[1.0, 1.0], [1.0, -1.0]]))
        print(grid.points[index])
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 100)
        grid.show(model, 0.5)


        Id1, Id0 = grid.works(model, 0.5)
        index, list_inds = grid.median(indices=Id0, return_indices=True)

        print(grid.points[index])
        print(list_inds)

def separation_test(i):
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 100)
        grid.show(model, 0.5)


        Id1, Id0 = grid.works(model, 0.5)
        print(grid.points[grid.median(indices=Id0)])
        #params, errIIsep1, errIIsep2 = SeparateAndCut(grid, Id1, Id0, 0.0)
        params = SeparateAndExplore(grid, Id0, [0.0, 0.0], [1.0, 1.0])

        print(params)

def centergrid_test(i):
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.CenteredGrid([0.0, 0.0], [1.0, 1.0], 1000, 10, model=model, inv_detection=True)
        grid.show(model, 0.5)


def ada_test(i):
    if(i == 2):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        params, grid = AdaManifPOST([0.0, 0.0], [1.0, 1.0], SeparateAndExplore, model, 1000)
        #print(grid.points.shape)
        #print(grid.N)
        grid.show(model, 0.5)
        print(params)
        print(grid.points)

def sphere_test(i):
    if(i == 0):
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 5)
        result = ExhaustiveDiminisher(grid, np.arange(len(grid.points)))
        print(result)
        vor = Voronoi(grid.points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',line_width=2, line_alpha=0.6, point_size=2)
        plt.show()
    elif(i == 1):
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 500)
        centers, _ = ExhaustiveDiminisher(grid, np.arange(len(grid.points)))
        print("exhaustive success")
        centers = np.array(centers)
        vor = Voronoi(grid.points)
        print("voronoi success")
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',line_width=2, line_alpha=0.6, point_size=2)
        #print(centers)
        plt.scatter(centers[:, 0], centers[:, 1])
        plt.show()
    elif(i == 2):
        grid = tools.std_grids.LowDiscrepancyGrid([0.0]*8, [1.0]*8, [2, 3, 5, 7, 11, 13, 17], 50)
        #centers, _ = ExhaustiveDiminisher(grid, np.arange(len(grid.points)))
        #print("exhaustive success")
        #centers = np.array(centers)
        vor = Voronoi(grid.points)
        print("voronoi success")
        print(len(vor.vertices))
    elif(i == 3):
        grid = tools.std_grids.LowDiscrepancyGrid([0.0]*2, [1.0]*2, [2], 500)
        vor = Voronoi(grid.points)
        print(vor.vertices)
        c, r = MLHS(grid, [0.0]*2, [1.0]*2)
        circle1 = plt.Circle(c, r, color='b', fill=False)
        print(c, r)
        points = np.array(grid.points)
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',line_width=2, line_alpha=0.6, point_size=2)
        plt.scatter(points[:,0], points[:,1])
        plt.gca().add_patch(circle1)
        plt.show()

def rect_test(i):
    if(i == 0):
        Compute_LECH([[0.1, 0.4], [0.6, 0.8]], [0.0, 0.0], [1.0, 1.0])

def mlhc_test(i):
    if(i == 0):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 5)
        grid.show(model, 0.5)


        Id1, Id0 = grid.works(model, 0.5)
        print(grid.points[grid.median(indices=Id0)])
        #params, errIIsep1, errIIsep2 = SeparateAndCut(grid, Id1, Id0, 0.0)
        params = MLHC(grid, np.array([0.0, 0.0]), np.array([1.0, 1.0]), Id0)

        print(params)
    elif(i == 1):
        model = tools.manifolds.Atlas([tools.manifolds.Cube([0.0, 0.1], [1.0, 0.6]), tools.manifolds.Cube([0.3, 0.0], [0.7, 1.0])])
        grid = tools.std_grids.LowDiscrepancyGrid([0.0, 0.0], [1.0, 1.0], [2], 100)
        grid.show(model, 0.5)


        Id1, Id0 = grid.works(model, 0.5)
        print(grid.points[grid.median(indices=Id0)])
        #params, errIIsep1, errIIsep2 = SeparateAndCut(grid, Id1, Id0, 0.0)
        params = MLHC(grid, np.array([0.0, 0.0]), np.array([1.0, 1.0]), Id0)

        print(params)

def hull_test(i):
    if(i == 0):
        a, b, n = [0.0, 0.0], [1.0, 1.0], 10
        grid = tools.std_grids.LowDiscrepancyGrid(a, b, [2], n)
        print(grid.points[5])
        print(tools.manifolds.Cube(a, b, index=5, grid=grid))

        model = tools.manifolds.Cube(a,b)
        grid.show(model, 0.5)
    elif(i == 1):
        a, b, n = [0.0, 0.0], [1.0, 1.0], 100
        grid = tools.std_grids.LowDiscrepancyGrid(a, b, [2], n)
        print(grid.points[50])
        print(tools.manifolds.Cube(a, b, index=50, grid=grid))

        model = tools.manifolds.Cube(a,b)
        grid.show(model, 0.5)


