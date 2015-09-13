# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:57:15 2015

@author: xerol
"""

import numpy as np
import interpolate as ip

class mc_exp(object):
    
    def __init__(self, number_of_particles=1e5, initial_energy=140.5):
        
        for _ in [number_of_particles,initial_energy]:
            try:
                float(_)
            except:
                print("{0} ist keine gültige Zahl.".format(_))
        
        self.startcount = number_of_particles
        self.startenergy = initial_energy
        
        self.initialize_particles()
        
    def initialize_particles(self):
        self.particles = np.ones((self.startcount,11))
        
        self.free_water = ip.interpolate("CrossSectWasser.txt")
        self.free_water.set_name(1,"scatter")
        self.free_water.set_name(2,"photo")
        
        self.free_lead = ip.interpolate("CrossSectBlei.txt")
        self.free_lead.set_name(1,"photo")
        
        self.particles[:,6] = self.startenergy
        self.particles[:,8] = self.free_water.interpolate(self.particles[:,6],
        "scatter")
        self.particles[:,9] = self.free_water.interpolate(self.particles[:,6],
        "photo")
        self.particles[:,7] = np.sum(self.particles[:,8:10],axis=1)
        self.particles[:,0:6] = 0
        
    def temp_angles(self, size):
        return np.append(2 * np.random.rand(size,1) - 1,
                         np.random.rand(size,1)*2*np.pi,axis=1)
                         
    def scatter_energy(self):
        self.particles[:,6] = self.particles[:,6] / (1 + (self.particles[:,6]/511.) * (1 - ))

    def move(self):
        pass