# -*- coding: utf-8 -*-
"""
Testfile für die particle-Klasse.
"""

import mc_exp as mc
import matplotlib.pyplot as plt
import numpy as np

print("Beginne Simulation, erzeuge Startarrays...")
a = mc.mc_exp(1e7, W=.99)
b = a.particles
a.poll_1()

print("Beginne Bewegung in Wasser, Fortschritt\n0%")
while b.count:
    a.move_particles()
    print("{0}%".format((1 - np.sum(a.water_mask)/float(len(a.water_mask)))*100))

a.poll_2()
a.cull_particles()
a.move_to_coll()
a.poll_3()

print("Starte Passage durch Kollimator...")
a.lead_length(5e3)

a.poll_4()
a.plot()