# -*- coding: utf-8 -*-
"""
Testfile für die particle-Klasse.
"""

import mc_exp as mc
import matplotlib.pyplot as plt

a = mc.mc_exp(1e6)
b = a.particles

while np.any(a.water_mask * (b.energy > 1e-3)):
    a.move_particles()