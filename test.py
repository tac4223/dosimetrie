# -*- coding: utf-8 -*-
"""
Testfile für die particle-Klasse.
"""

import mc_exp as mc
import matplotlib.pyplot as plt
import numpy as np

print("Beginne Simulation, erzeuge Startarrays...")
a = mc.mc_exp(1e7, W=.9)
b = a.particles

print("Beginne Bewegung in Wasser, Fortschritt\n0%")
while b.count:
    a.move_particles()
    print("{0}%".format((1 - np.sum(a.water_mask)/float(len(a.water_mask)))*100))

print("Anteil an Teilchen die die Kugel verlassen: {0}%".format(np.round(np.sum(b.weight)/a.init_count*100,2)))

a.cull_particles()
a.move_to_coll()

print("Anteil der den Kollimator trifft: {0}%".format(np.round(a.colhit_ratio*100.,2)))

a.lead_length(1e7)