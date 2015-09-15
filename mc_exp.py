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
        self.count = number
        self.coords = np.zeros((number,3))
        self.direction = np.zeros((number,3))
        self.energy = np.ones(number) * initial_energy
        self.weight = np.ones(number)
        self.mu = np.zeros(number)
        self.phi = 1*self.mu
        self.scatter = np.zeros(number)
        self.photo = 1*self.scatter
        self.total_x = 1*self.scatter
        self.p_scatter = 1*self.scatter
        self.p_photo = 1*self.scatter

    def klein_nishina(self, mu, energy):
        """
        Gibt den Wert der KN-Formel zu gegebenem µ und E zurück. Vorfaktoren
        weggelassen, das wäre nur unnötige Rechnerei.
        """
        y = energy/511.
        return 1./(1 + y * (1 - mu))**2 * (1 + mu**2 + ((y**2)*(1 - mu)**2)/\
        (1 + y*(1 - mu)))

    def guess_kn(self, count):
        """
        Rät µ und klein_nishina(µ), zwecks Verwendung beim Auswürfeln von µ per
        Verwerfungsmethode. Da µ zwischen -1 und 1 liegt, der Funktionswert
        der Klein-Nishina-Formel aber zwischen 0 und 2, lassen sich beide
        auf einmal würfeln und anschließend eine Spalte um -1 verschieben.
        """
        return 2 * np.random.rand(count,2) - np.array([1,0])

    def get_angles(self):
        """
        Beginnt mit guess_kn() in maximaler Größe. Anschließend werden
        sukzessive alle Zeilen die nicht als korrekte Realisierung der
        Zufallsvariablen in Frage kommen neu gewürfelt.
        Zuletzt wird self.mu mit den gefundenen Werten angepasst, und self.phi
        gewürfelt.
        """

        initial_guess = self.guess_kn(self.count)

        while np.any((self.klein_nishina(initial_guess[:,0],self.energy) <
        initial_guess[:,1])[0]):
            mask = self.klein_nishina(initial_guess[:,0],self.energy) < \
            initial_guess[:,1]
            initial_guess[mask] = self.guess_kn(len(self.energy[mask]))

        self.mu = initial_guess[:,0]
        self.phi = np.random.rand(self.count)*2*np.pi

    def E_scatter(self):
        """
        Aktualisiert die Teilchenenergien nach einem Stoß.
        """
        self.energy = self.energy / (1 + (self.energy/511) * (1 - self.mu))

    def mean_free(self):
        """
        Spuckt eine Runde freie Weglängen aus, basierend auf den derzeitigen
        Werten für die Wirkungsquerschnittssumme.
        """
        return 1./self.total_x * np.log(np.random.rand(len(self.total_x)))

    def move(self):
        self.coords += self.direction * self.mean_free()

    def interact(self):
        self.photo_mask = np.logical_not(self.scatter_mask)

        self.weight[self.photo_mask] *= self.p_photo[self.photo_mask]

        self.E_scatter(self.scatter_mask)

    def direction(self):
        pass

class mc_exp(object):
    """
    Enthält das eigentliche Experiment, inklusive des Vorwissens über den
    Versuchsaufbau (Wasserkugel, Bleikollimator...). Zudem findet das Input
    Sanitizing für die anderen Klassen statt.

    Variablen:
    self.init_E: Anfangsenergie die allen Teilchen mitgegeben wird.

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

        self.particles = particles(number_of_particles,self.init_E)

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

