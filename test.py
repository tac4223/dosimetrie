# -*- coding: utf-8 -*-
"""
Testfile für die particle-Klasse.
"""

import mc_exp as mc
import matplotlib.pyplot as plt

a = mc.mc_exp(50)
b = a.particles

while np.any(a.water_mask):
    a.move_particles()
print(len(b.energy))
print(b.weight)