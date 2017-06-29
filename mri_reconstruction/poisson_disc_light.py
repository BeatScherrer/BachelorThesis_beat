# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:11:45 2017

@author: beats
"""

from math import sin
from math import cos
from math import sqrt
from math import ceil
from math import pi

from random import randint
from random import uniform

import numpy as np

from datastructures import RandomQueue
from enhanced_grid import int_point_2d
from enhanced_grid import Grid2D
from enhanced_grid import ListGrid2D

def sqr_dist((x0, y0), (x1, y1)):
	return (x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0)

def dist((x0, y0), (x1, y1)):
	u = (x1 - x0)
	v = (y1 - y0)
	
	return sqrt(u*u + v*v)

def rand(n):
	return randint(0, n - 1)

def make_image_grid_from_radius_grid_2(r_grid, min, max):
	d = max - min
	i_grid = Grid2D(r_grid.dims, 0)
	
	for index in r_grid.index_iter():
		i_grid[index] = (r_grid[index] - min) / d
	
	return i_grid

def poisson_circle((shape)):
	h = shape[1]
	w = shape[2]
	r_grid = Grid2D((w, h))
	center = (w/2, h/2)

	for index in r_grid.index_iter():
            r_grid[index] = (dist(index, center)/5) + 0.1 # avoid 0 radius!

	p = sample_poisson(w, h, r_grid, 30)
    
	mask = np.zeros((w,h))
	for item in p:
            mask[item]=1
	return mask

def sample_poisson(width, height, r_grid, k):	
	#Convert rectangle (the one to be sampled) coordinates to 
	# coordinates in the grid.
	def grid_coordinates((x, y)):
		return (int(x*inv_cell_size), int(y*inv_cell_size))
	
	# Puts a sample point in all the algorithm's relevant containers.
	def put_point(p):
		process_list.push(p)
		sample_points.append(p)  
		grid[grid_coordinates(p)].append(p)

	# Generates a point randomly selected around
	# the given point, between r and 2*r units away.
	def generate_random_around((x, y), r):
		rr = uniform(r, 2*r)
		rt = uniform(0, 2*pi)
		
		return int(rr*sin(rt) + x), int(rr*cos(rt) + y)
		
	# Is the given point in the rectangle to be sampled?
	def in_rectangle((x, y)):
		return 0 <= x < width and 0 <= y < height
	
	def in_neighbourhood(p, r):
		gp = grid_coordinates(p)
		r_sqr = r*r
		
		for cell in grid.square_iter(gp, 2):
			for q in cell:
				if sqr_dist(q, p) <= r_sqr:
					return True
		return False

	r_min, r_max = r_grid.min_max()
	
	#Create the grid
	cell_size = r_max/sqrt(2)
	inv_cell_size = 1 / cell_size	
	r_max_sqr = r_max*r_max
	
	grid = ListGrid2D(int_point_2d((ceil(width/cell_size),
		ceil(height/cell_size))))
		
	process_list = RandomQueue()
	sample_points = []	
	
	#generate the first point
	put_point((int(rand(width)), int(rand(height))))
	
	#generate other points from points in queue.
	while not process_list.empty():
		p = process_list.pop()
		r = r_grid[int_point_2d(p)]
		
		for i in xrange(k):			
			q = generate_random_around(p, r)
			if in_rectangle(q) and not in_neighbourhood(q, r):
					put_point(q)
	
	return sample_points