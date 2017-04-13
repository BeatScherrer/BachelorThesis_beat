# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:28:57 2017

This Algorithm is based of The Paper 'Fast Poisson Disk Sampling in Arbitrary Dimensions'
from Robert Bridson (University of Columbia)

http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

@author: beats (sbeat@student.ethz.ch)
"""
import numpy as np
import random
from itertools import product


class Grid:
    
    def __init__(self, r, *shape, k=30):
        self.r = r
        self.shape = shape
        self.k = k
        self.dim = len(shape)
        
        self.cell_size = r/np.sqrt(self.dim)
        
        self.rows = int(self.shape[0]/self.cell_size)+1
        self.cols = int(self.shape[1]/self.cell_size)+1
        #initialize grid as dict
        self.grid = {num:-1 for num in product(*(range(self.cols),range(self.rows)))}
        
        #initialize with list for easier appending
        self.samples = []
        self.active = []
    
    def clear(self):
        '''
        resets the grid, active points and sample points
        '''
        self.samples = []
        self.active = []
        
    def poisson(self):
        self.clear()
        
        # Step 1: pick random point chosen uniformly from domain
        x0 = np.array([random.uniform(0,self.shape[0]), random.uniform(0,self.shape[1])])
        self.samples.append(x0)
        self.active.append(0)
        self.update(x0, 0)
        
        # Step 2:
        while self.active:
            i = random.choice(self.active)
            point = self.samples[i]
            
        
    def make_points(self):
        
    def upgrade_grid(self, point, index):
        '''
        updates the grid with the new point
        '''
        self.grid[point] = index
        
    
    def check(self):
        '''
        Check if point is within distance r of existing samples
        (using background grid to only test nearby samples).
        '''
        return
        
    def distance(self):
        return
        
        
        
        
    
