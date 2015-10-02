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
    count: Anzahl an Teilchen, die aktuell von Interesse sind.
    coords: Globaler Ortsvektor
    directions: Globaler Richtungsvektor
    energy: Teilchenenergie
    weight: Teilchenwichtung
    mu: cos(Theta) für jedes Teilchen
    phi: Phi für alle Teilchen
    scatter: Streuquerschnitt, wird von extern verändert.
    photo: Querschnitt für Photoabsorption, von extern verändert.
    total_x: Summe beider Querschnittswerte.
    p_photo: Absorptionswahrscheinlichkeit.

    Funktionen:
    interact:
    E_scatter: Passt nach Streuung die Teilchenenergie an.

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
        self.p_photo = 1*self.scatter



    def interact(self,particle_mask = None):
        """
        Dient als Einstiegsfunktion für die Teilcheninteraktion. Übergibt allen
        anderen Funktionen die passenden Parameter, ruft sie in der richtigen
        Reihenfolge auf und wendet sie nur auf die in particle_mask gegebenen
        Teilchen an.
        """
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count,dtype=bool)
        else:
            self.count = np.sum(particle_mask)

        photo_mask = (np.random.rand(len(particle_mask)) < self.p_photo) * \
            particle_mask
        self.weight[photo_mask] *= (1-self.p_photo[photo_mask])

        self.get_angles(particle_mask)
        self.get_direction(particle_mask)
        self.E_scatter(particle_mask)
        self.move(particle_mask)

    def get_angles(self, particle_mask = None):
        """
        Beginnt mit guess_kn() in maximaler Größe. Anschließend werden
        sukzessive alle Zeilen die nicht als korrekte Realisierung der
        Zufallsvariablen in Frage kommen neu gewürfelt.
        Zuletzt wird self.mu mit den gefundenen Werten angepasst, und self.phi
        gewürfelt.
        """
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count,dtype=bool)

        initial_guess = self.guess_kn(self.count)

        while np.any(self.klein_nishina(initial_guess[:,0],
                            self.energy[particle_mask]) < initial_guess[:,1]):
                invalid = self.klein_nishina(initial_guess[:,0],
                         self.energy[particle_mask]) < initial_guess[:,1]
                initial_guess[invalid] = self.guess_kn(np.sum([invalid]))

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
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count,dtype=bool)

        K_tilde = np.zeros((self.count,3,3))
        local_direction = np.zeros((self.count,3))

        local_direction[:,0] = self.mu[particle_mask]
        local_direction[:,1] = np.sqrt(1 - self.mu[particle_mask]**2) * \
        np.cos(self.phi[particle_mask])
        local_direction[:,2] = np.sqrt(1 - self.mu[particle_mask]**2) * \
        np.sin(self.phi[particle_mask])

        x_mask = (np.abs(self.direction[:,0][particle_mask]) <= 1e8) * \
            (np.abs(self.direction[:,1][particle_mask]) >= 1e8)
        y_mask = (np.abs(self.direction[:,1][particle_mask]) == 0)
        else_mask = np.logical_not(np.logical_or(x_mask,y_mask))

        K_tilde[:,:,0] = self.direction[particle_mask]

        K_tilde[:,:,1][else_mask] = np.transpose(
            np.array([1./self.direction[:,0][particle_mask][else_mask],
              -1./self.direction[:,1][particle_mask][else_mask],
              np.zeros(np.sum(else_mask))]))

        K_tilde[:,:,1] /= np.reshape(
            np.sqrt(np.sum(K_tilde[:,:,1]**2,1)),(-1,1))

        K_tilde[:,:,1][x_mask] = [1,0,0]
        K_tilde[:,:,1][y_mask] = [0,1,0]

        K_tilde[:,:,2] = np.reshape(np.cross(K_tilde[:,:,0],
            K_tilde[:,:,1]),(-1,3))

        self.direction[particle_mask] = np.sum(K_tilde *
            np.reshape(local_direction,(-1,3,1)),axis=1)

    def E_scatter(self, particle_mask = None):
        """
        Aktualisiert die Teilchenenergien nach einem Stoß.
        """
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count) == 1

        self.energy[particle_mask] /= (1 + (self.energy[particle_mask]/.511) *\
            (1 - self.mu[particle_mask]))

        for _ in vars(self):
            try:
                vars(self)[_] = vars(self)[_][self.energy > 1e-3]
            except:
                pass

    def move(self, particle_mask = None):
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count,dtype=bool)

        self.coords[particle_mask] += self.direction[particle_mask] * \
            self.mean_free(particle_mask)

    def mean_free(self, particle_mask = None):
        """
        Spuckt eine Runde freie Weglängen aus, basierend auf den derzeitigen
        Werten für die Wirkungsquerschnittssumme.
        """
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count,bool)

        return np.reshape(-1./self.total_x[particle_mask] * \
            np.log(np.random.rand(self.count)),(-1,1))

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

        self.water_mask = np.ones(self.init_count,dtype=bool)

        self.update_xsect()
        self.initial_move()

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


    def initial_move(self):
        """
        Dient nur dazu, die Teilchen direkt nach der Erzeugung auf um eine
        freie Weglänge zuföllig um die Quelle verteilte Positionen zu schießen.
        """
        angles = 2*np.random.rand(self.particles.count,2) - np.array([1,0])
        angles[:,1] *= np.pi

        self.particles.mu = angles[:,0]

        self.particles.direction[:,0] = angles[:,0]
        self.particles.direction[:,1] = np.sqrt(1 - angles[:,0]**2) * \
            np.cos(angles[:,1])
        self.particles.direction[:,2] = np.sqrt(1 - angles[:,0]**2) * \
            np.sin(angles[:,1])

        self.particles.move()
        self.particles.E_scatter()


    def move_particles(self):
        """
        Bewegt die Teilchen entsprechend der experimentellen Parameter weiter.
        """
        if np.any(self.water_mask) == True:
            self.water_mask = self.water_mask[self.particles.energy > 1e-3]
            self.update_xsect()
            self.particles.interact(self.water_mask)
        else:
            print("All particles outside of water sphere.")
