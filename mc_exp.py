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
    E_scatter: Passt nach Streuung die Teilchenenergie an. Als Parameter muss
    boolean-Maske übergeben werden, die angibt bei welchen Teilchen eine
    Streuung stattfand.
    """
    def __init__(self, number=1e5, initial_energy=.1405):
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

        self.K_tilde = np.zeros((self.count,3,3))
        self.local_direction = np.zeros((self.count,3))

    def interact(self,particle_mask = None):
        """
        Dient als Einstiegsfunktion für die Teilcheninteraktion. Übergibt allen
        anderen Funktionen die passenden Parameter, ruft sie in der richtigen
        Reihenfolge auf und wendet sie nur auf die in particle_mask gegebenen
        Teilchen an.
        """
        if particle_mask == None:
            particle_mask = np.ones(self.count)

        self.count = len(self.energy[particle_mask])
        photo_mask = np.random.rand(self.count) < self.p_photo[particle_mask]

        self.weight[particle_mask][photo_mask] *= \
            self.p_scatter[particle_mask][photo_mask]

        self.get_angles()
        self.get_direction()
        self.E_scatter(particle_mask)

    def get_angles(self, particle_mask = None):
        """
        Beginnt mit guess_kn() in maximaler Größe. Anschließend werden
        sukzessive alle Zeilen die nicht als korrekte Realisierung der
        Zufallsvariablen in Frage kommen neu gewürfelt.
        Zuletzt wird self.mu mit den gefundenen Werten angepasst, und self.phi
        gewürfelt.
        """
        if particle_mask == None:
            particle_mask = np.ones(self.count)

        initial_guess = self.guess_kn(self.count)

        while np.any((self.klein_nishina(initial_guess[:,0],self.energy[particle_mask]) <
            initial_guess[:,1])[0]):
                mask = self.klein_nishina(initial_guess[:,0],self.energy[particle_mask]) < \
                    initial_guess[:,1]
                initial_guess[mask] = self.guess_kn(len(self.energy[particle_mask][mask]))

        self.mu[particle_mask] = initial_guess[:,0]
        self.phi[particle_mask] = np.random.rand(self.count)*2*np.pi

    def guess_kn(self, count):
        """
        Rät µ und klein_nishina(µ), zwecks Verwendung beim Auswürfeln von µ per
        Verwerfungsmethode. Da µ zwischen -1 und 1 liegt, der Funktionswert
        der Klein-Nishina-Formel aber zwischen 0 und 2, lassen sich beide
        auf einmal würfeln und anschließend eine Spalte um -1 verschieben.
        """
        return 2 * np.random.rand(count,2) - np.array([1,0])

    def klein_nishina(self, mu, energy):
        """
        Gibt den Wert der KN-Formel zu gegebenem µ und E zurück. Vorfaktoren
        weggelassen, das wäre nur unnötige Rechnerei.
        """
        y = energy/.511
        return 1./(1 + y * (1 - mu))**2 * (1 + mu**2 + ((y**2)*(1 - mu)**2)/\
            (1 + y*(1 - mu)))

    def get_direction(self, particle_mask = None):
        """
        Errechnet basierend auf den ausgewürfelten Werten für phi und mu einen
        neuen Richtungsvektor für jedes Teilchen. Hierzu wird die Drehmatrix K~
        für alle ermittelten lokalen Richtungsvektoren aufgestellt und die
        Streurichtungsvektoren damit transformiert.
        """
        if particle_mask == None:
            particle_mask = np.ones(self.count)

        self.local_direction[particle_mask][:,0] = self.mu[particle_mask]
        self.local_direction[particle_mask][:,1] = np.sqrt(1 - self.mu[particle_mask]**2) * np.cos(self.phi[particle_mask])
        self.local_direction[particle_mask][:,2] = np.sqrt(1 - self.mu[particle_mask]**2) * np.sin(self.phi[particle_mask])

        self.x_mask = (np.abs(self.direction[particle_mask][:,0]) == 0) * \
            (np.abs(self.direction[particle_mask][:,1]) == 0)

        self.y_mask = (np.abs(self.direction[particle_mask][:,1]) == 0)
        else_mask = np.logical_not(np.logical_or(self.x_mask,self.y_mask))


        self.K_tilde[particle_mask][:,:,0] = self.direction[particle_mask]
        self.K_tilde[particle_mask][:,:,1][self.x_mask] = [1,0,0]
        self.K_tilde[particle_mask][:,:,1][self.y_mask] = [0,1,0]

        self.K_tilde[particle_mask][:,:,1][else_mask] = np.transpose(
            np.array([1./self.direction[particle_mask][:,0],-1./self.direction[particle_mask][:,1],
              np.zeros(self.count)]))

        self.K_tilde[particle_mask][:,:,1] /= np.reshape(
            np.sqrt(np.sum(self.K_tilde[particle_mask][:,:,1]**2,1)),(-1,1))

        self.K_tilde[particle_mask][:,:,2] = np.reshape(np.cross(self.K_tilde[particle_mask][:,:,0],
            self.K_tilde[particle_mask][:,:,1]),(-1,3))

        self.direction[particle_mask] = np.sum(self.K_tilde[particle_mask] * np.reshape(self.local_direction[particle_mask],
            (-1,3,1)),axis=1)

    def E_scatter(self, particle_mask = None):
        """
        Aktualisiert die Teilchenenergien nach einem Stoß.
        """
        if particle_mask == None:
            particle_mask = np.ones(self.count)

        self.energy = self.energy[particle_mask] / (1 + (self.energy[particle_mask]/.511) *\
            (1 - self.mu[particle_mask]))

    def move(self, particle_mask = None):
        if particle_mask == None:
            particle_mask = np.ones(self.count)

        self.coords[particle_mask] += self.direction[particle_mask] * \
            np.reshape(self.mean_free(particle_mask),(-1,1))

    def mean_free(self, particle_mask = None):
        """
        Spuckt eine Runde freie Weglängen aus, basierend auf den derzeitigen
        Werten für die Wirkungsquerschnittssumme.
        """
        if particle_mask == None:
            particle_mask = np.ones(self.count)
        return -1./self.total_x[particle_mask] * np.log(np.random.rand(self.count))

    def initial_move(self, particle_mask = None):
        """
        Dient nur dazu, die Teilchen direkt nach der Erzeugung auf um eine
        freie Weglänge zuföllig um die Quelle verteilte Positionen zu schießen.
        """
        if particle_mask == None:
            particle_mask = np.ones(self.count)
        angles = 2*np.random.rand(self.count,2) - np.array([1,0])
        angles[:,1] *= np.pi
        self.mu = angles[:,0]

        self.direction[:,0] = angles[:,0]
        self.direction[:,1] = np.sqrt(1 - angles[:,0]**2) * np.cos(angles[:,1])
        self.direction[:,2] = np.sqrt(1 - angles[:,0]**2) * np.sin(angles[:,1])

        self.move()
        self.E_scatter()

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
    def __init__(self, number_of_particles=1e5, initial_energy=0.1405):
        """
        Input sanitizing, Erstellen der interpolate- und particle-Instanzen.
        Dichte von Wasser und Blei wird absichtlich um eine Größenordnung zu
        niedrig angegeben, um quasi implizit auf 1/mm für den
        Wirkungsquerschnitt umzurechnen.
        """
        for _ in [number_of_particles,initial_energy]:
            try:
                float(_)
            except:
                print("{0} ist keine gültige Zahl.".format(_))

        self.init_E = initial_energy
        self.init_count = number_of_particles

        self.particles = particles(number_of_particles,self.init_E)

        self.water = ip.interpolate("CrossSectWasser.txt",.1)
        self.water.set_name(1,"scatter")
        self.water.set_name(2,"photo")
        self.lead = ip.interpolate("CrossSectBlei.txt",1.134)
        self.lead.set_name(1,"photo")

        self.water_mask = np.ones(self.init_count)

        self.update_xsect()
        self.particles.initial_move()

    def update_xsect(self):
        """
        Auf Basis der interpolate-Instanzen werden die Wirkungsquerschnitte in
        der particle-Instanz neu geschrieben. Wasserwerte solang innerhalb der
        Kugel, Bleiwerte außerhalb.
        """
        self.new_lead = self.water_mask
        self.water_mask = np.sum(self.particles.coords**2,axis=1) <= 1e4
        self.new_lead = self.new_lead * np.logical_not(self.water_mask)

        self.particles.scatter[self.water_mask] = self.water.interpolate(
        self.particles.energy[self.water_mask],"scatter")

        self.particles.scatter[self.new_lead] = 0

        self.particles.photo[self.water_mask] = self.water.interpolate(
        self.particles.energy[self.water_mask],"photo")

        self.particles.photo[self.new_lead] = self.water.interpolate(
        self.particles.energy[self.new_lead],"photo")

        self.particles.total_x = self.particles.photo + self.particles.scatter
        self.particles.p_photo = self.particles.photo / self.particles.total_x
        self.particles.p_scatter = self.particles.scatter /\
            self.particles.total_x

    def move_particles(self):
        """
        Bewegt die Teilchen entsprechend der experimentellen Parameter weiter.
        """
        self.update_xsect()
        self.particles.interact(self.water_mask)
        self.particles.move(self.water_mask)

