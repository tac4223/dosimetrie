# -*- coding: utf-8 -*-
"""

"""
import numpy as np

class interpolate(object):
    
    def __init__(self, query_column, data, headersize):
        self.data = np.loadtxt(data,skiprows=headersize)
        self.qcolumn = query_column
        
    def interpolate(self, query_energy):
        self.qenergy = query_energy
        
        self.find_neighbours()
        return(self.find_xsect())
        
    def find_neighbours(self):
        self.mask_low = self.data[:,0] <= self.qenergy
        self.mask_high = self.data[:,0] >= self.qenergy
        
        self.nrgy = [self.data[:,0][self.mask_low][-1],
                     self.data[:,0][self.mask_high][0]]
                     
        self.xsect = [self.data[:,1][self.mask_low][-1],
                     self.data[:,1][self.mask_high][0]]
                     
    def find_xsect(self):
        if self.xsect[0] == self.xsect[1]:
            return self.xsect[0]
        else:
            return self.xsect[0] + (self.xsect[1] - self.xsect[0]) * \
            (self.qenergy - self.nrgy[0])/(abs(self.nrgy[1] - self.nrgy[0]))