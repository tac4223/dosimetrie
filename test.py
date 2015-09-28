# -*- coding: utf-8 -*-
"""
Testfile für die particle-Klasse.
"""

import mc_exp as mc
import matplotlib.pyplot as plt

test = mc.mc_exp(1e6)
test.particles.get_angles()
test.particles.get_direction()