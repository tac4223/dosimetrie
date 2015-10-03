# -*- coding: utf-8 -*-
"""
Testfile für die particle-Klasse.
"""

import mc_exp as mc
import matplotlib.pyplot as plt
import numpy as np

a = mc.mc_exp(1e7,W=0.01)
b = a.particles

while b.count:
    a.move_particles()
    print((1 - np.sum(a.water_mask)/float(len(a.water_mask)))*100)

a.cull_particles()
a.move_to_coll()

print(np.sum(b.weight)/a.init_count)