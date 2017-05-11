from __future__ import with_statement


from poisson_disk import *
from enhanced_grid import *

w = h = 256
min_radius = 4
max_radius = 20

def dist((x0, y0), (x1, y1)):
	u = (x1 - x0)
	v = (y1 - y0)
	
	return sqrt(u*u + v*v)

	
def make_image_grid_from_radius_grid_2(r_grid, min, max):
	d = max - min
	i_grid = Grid2D(r_grid.dims, 0)
	
	for index in r_grid.index_iter():
		i_grid[index] = (r_grid[index] - min) / d
	
	return i_grid

	
def poisson_circle_demo():
	w = h = 512
	r_grid = Grid2D((w, h))
	center = (w/2, h/2)

	for index in r_grid.index_iter():
            r_grid[index] = (dist(index, center)) + 0.1 # avoid 0 radius!

	p = sample_poisson(w, h, r_grid, 30)

	g = points_to_grid(p, (w, h))

	convert_enhanced_grid_to_png(g, 'poisson_circle_image.png', True)

	i_grid = make_image_grid_from_radius_grid_2(r_grid, 0.1, (sqrt(w*w/2)))
    
	return p,g,i_grid
	
    
print "Demo"

p,g,i_grid = poisson_circle_demo()
