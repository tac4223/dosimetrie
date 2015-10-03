# -*- coding: utf-8 -*-
"""
Testfile für die particle-Klasse.
"""

import mc_exp as mc
import matplotlib.pyplot as plt
import numpy as np

a = mc.mc_exp(1e5,W=1. - 1e-8)
b = a.particles

while b.count:
    a.move_particles()
print(len(b.energy))
print(b.weight)