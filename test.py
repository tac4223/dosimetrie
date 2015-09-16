# -*- coding: utf-8 -*-
"""
Testfile für die particle-Klasse.
"""

import mc_exp as mc
import matplotlib.pyplot as plt

test = mc.mc_exp(5)
pars = [test.particles.coords, test.particles.direction, test.particles.scatter, test.particles.photo]
for _ in pars:
    print(_)