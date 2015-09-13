# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:57:15 2015

@author: xerol
"""

import numpy as np
import interpolate as ip

class particles(object):
    """
    Klasse die praktische Zusammenfassung aller direkt partikelbezogenen
    Eigenschaften bietet. Darüber hinaus werden die grundsätzlichen
    Möglichkeiten zur Wechselwirkung und Bewegung bereitgestellt.

    Variablen:
    self.coords: Globaler Ortsvektor
    self.directions: Globaler Richtungsvektor
    self.energy: Teilchenenergie
    self.weight: Teilchenwichtung
    self.mu: cos(Theta) für jedes Teilchen
    self.phi: Phi für alle Teilchen
    self.scatter: Streuquerschnitt, wird von extern verändert.
    self.photo: Querschnitt für Photoabsorption, von extern verändert.
    self.total_x: Summe beider Querschnittswerte.
    self.p_scatter: Streuwahrscheinlichkeit.
    self.p_photo: Absorptionswahrscheinlichkeit.

    Funktionen:
    temp_roll_angles: Würfelt Streuwinkel aus.
    E_scatter: Passt nach Streuung die Teilchenenergie an. Als Parameter muss
    boolean-Maske übergeben werden, die angibt bei welchen Teilchen eine
    Streuung stattfand.
    """
    def __init__(self, number=1e5, initial_energy=140.5):
        """
        Fixiert alle aus den zwei Initialwerten abzuleitenden Werte zunächst
        mal in Objekteigenschaften, hier passiert eigentlich nichts
        spannendes. Alle Partikel werden bei [0,0,0] erzeugt, mit Richtung
        [0,0,0] und ohne berechnete Wirkungsquerschnitte oder Streuwinkel.
        """
        self.coords = np.zeros((number,3))
        self.direction = np.zeros((number,3))
        self.energy = np.ones((number,1)) * initial_energy
        self.weight = np.ones((number,1))
        self.mu = np.zeros((number,1))
        self.phi = 1*self.mu
        self.scatter = np.zeros((number,1))
        self.photo = 1*self.scatter
        self.total_x = 1*self.scatter
        self.p_scatter = 1*self.scatter
        self.p_photo = 1*self.scatter

    def temp_roll_angles(self):
        """
        Ermöglicht bequeme Erzeugung von Streuwinkeln für alle Partikel. Hier
        muss noch nachgearbeitet werden, µ wird einfach nur gleichverteilt
        erzeugt! Energieabhängigkeit fehlt völlig, ich kapier die fucking
        Klein-Nishina-Formel noch nicht.
        """
        ###############################################
        self.mu = 2 * np.random.rand(self.count,1) - 1
        ### NOCH ZU ÄNDERN - NOCH ZU ÄNDERN - NOCH ZU Ä
        ###############################################

        self.phi = np.random.rand(self.count,1)*2*np.pi

    def E_scatter(self, mask):
        """
        Aktualisiert die Teilchenenergien nach einem Stoß. Maske gibt an welche
        Teilchen gestreut wurden.
        """
        self.energy[mask] = self.energy[mask] / (1 + (self.energy[mask]/511)
        * (1 - self.mu[mask]))

    def mean_free(self, size):
        """
        Spuckt eine Runde freie Weglängen aus, basierend auf den derzeitigen
        Werten für die Wirkungsquerschnittssumme.
        """
        return 1./self.total_x * np.log(np.random.rand(size,1))

class mc_exp(object):
    """
    Enthält das eigentliche Experiment, inklusive des Vorwissens über den
    Versuchsaufbau (Wasserkugel, Bleikollimator...). Zudem findet das Input
    Sanitizing für die anderen Klassen statt.

    Variablen:
    self.init_E: Anfangsenergie die allen Teilchen mitgegeben wird.
    self.count: Anzahl zu erzeugender Teilchen.

    Instanzen:
    self.water: interpolate-Instanz mit Wasserdaten.
    self.lead: interpolate-Instanz mit Bleidaten.
    self.particles: particles-Instanz.

    Funktionen:
    self.update_xsect: Überschreibt die Einträge für Querschnitte mit jenen
    für das jeweilig umgebende Material (Wasser innerhalb der Kugel, Blei außer-
    halb).
    """
    def __init__(self, number_of_particles=1e5, initial_energy=140.5):
        """
        Input sanitizing, Erstellen der interpolate- und particle-Instanzen.

        """
        for _ in [number_of_particles,initial_energy]:
            try:
                float(_)
            except:
                print("{0} ist keine gültige Zahl.".format(_))

        self.init_E = initial_energy
        self.count = number_of_particles

        self.particles = particles(self.count,self.init_E)

        self.water = ip.interpolate("CrossSectWasser.txt")
        self.water.set_name(1,"scatter")
        self.water.set_name(2,"photo")
        self.lead = ip.interpolate("CrossSectBlei.txt")
        self.lead.set_name(1,"photo")

        self.update_xsect()

    def update_xsect(self):
        """
        Auf Basis der interpolate-Instanzen werden die Wirkungsquerschnitte in
        der particle-Instanz neu geschrieben. Wasserwerte solang innerhalb der
        Kugel, Bleiwerte außerhalb.
        """
        self.water_mask = np.sum(self.particles.coords**2,axis=1) <= 1e4
        self.lead_mask = np.logical_not(self.water_mask)

        self.particles.scatter[self.water_mask] = self.water.interpolate(
        self.particles.energy[self.water_mask],"scatter")

        self.particles.scatter[self.lead_mask] = 0


        self.particles.photo[self.water_mask] = self.water.interpolate(
        self.particles.energy[self.water_mask],"photo")

        self.particles.photo[self.lead_mask] = self.water.interpolate(
        self.particles.energy[self.lead_mask],"photo")

        self.particles.total_x = self.particles.photo + self.particles.scatter
        self.particles.p_photo = self.particles.photo / self.particles.total_x
        self.particles.p_scatter = self.particles.scatter /\
        self.particles.total_x

