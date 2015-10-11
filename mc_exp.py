# -*- coding: utf-8 -*-
"""
Objekt zum einfachen Interpolieren von Querschnitten bei gegebener Energie.
Wird initialisiert mit Dateinamen einer Tabelle entsprechender Querschnitte.
Ferner müssen die einzelnen Spalten mit Namen belegt werden, um bei Tabellen
mit mehreren Spalten Verwechslungen auszuschließen.

Variablen
self.data: Enthält die Daten zwischen denen interpoliert werden soll.
self.colnames: Dictionary, das zwischen tatsächlich merkbaren Bezeichnungen und
den Spaltennummern vermittelt.

Funktionen
self.set_name(): Zuordnung von Spaltenzahlen und deren Bezeichnung.
self.interpolate(): Die eigentliche Interpolation. Wird von außen mit einem x-
Wert und einer Spaltenbezeichnung aufgerufen.
"""
import numpy as np

class interpolate(object):

    def __init__(self, data, density=1, headersize=3):
        """
        Schlichte Initialisierung. Als Data wird der Dateiname eines Textfiles
        übergeben, in dem die zu interpolierenden Daten enthalten sind.
        Density kann für Wirkungsquerschnitte übergeben werden, Werte werden
        damit multipliziert (default 1). Headersize gibt die Anzahl an
        Kopfzeilen an, die übersprungen werden müssen um zum Beginn der
        Zahlenwerte zu kommen.
        """
        self.data = np.loadtxt(data,skiprows=headersize)
        self.data[:,1::] *= density
        self.colnames = {}

    def set_name(self,column,name):
        """
        Sollte ausgeführt werden um von schlicht durchnummerierten Spalten zu
        sprechenden Namen zu kommen. Namen werden in self.interpolate() ver-
        wendet.
        """
        self.colnames[name] = column

    def interpolate(self, energy, y_col=1):
        """
        Sollte self.colnames leer sein, wird einfach nur die erste Spalte als
        x- und die zweite Spalte als y-Wert genommen zum Interpolieren.
        Falls self.colnames existiert, wird die korrekte Spalte entsprechend
        des angegebenen Namens ausgewählt.
        """
        if self.colnames:
            return np.interp(energy, self.data[:,0],
                             self.data[:,self.colnames[y_col]])
        return np.interp(energy, self.data[:,0],
                             self.data[:,1])

import interpolate as ip
import matplotlib.pyplot as plt

class particles(object):
    """
    Klasse die praktische Zusammenfassung aller direkt partikelbezogenen
    Eigenschaften bietet. Darüber hinaus werden die grundsätzlichen
    Möglichkeiten zur Wechselwirkung und Bewegung bereitgestellt. Teilchen
    werden zunächst an Ort [0,0,0] erzeugt, ohne Richtung oder Wirkungsquer-
    schnitte.

    Instanzvariablen:
    coords: Globaler Ortsvektor
    count: Anzahl an Teilchen, die aktuell von Interesse sind.
    direction: Globaler Richtungsvektor
    energy: Teilchenenergie
    min_energy: Teilchen mit Energie (MeV) unter diesem Wert werden gelöscht
    min_weight: Mindestens verbleibendes Restgewicht, unterhalb liegende
        Teilchen werden gelöscht.
    mu: cos(Theta) für jedes Teilchen
    p_photo: Absorptionswahrscheinlichkeit.
    phi: Phi für alle Teilchen
    photo: Querschnitt für Photoabsorption, von extern verändert.
    properties: Liste der Objektvariablen, exklusive Teilchenzahl. Wird von
        Funktion cleanup verwendet.
    scatter: Streuquerschnitt, wird von extern verändert.
    total_x: Summe beider Querschnittswerte.
    weight: Teilchenwichtung




    Funktionen:
    interact: Einstiegsfunktion, einzige Funktion die von außerhalb aufgerufen
        werden sollte.
    get_angles: Erzeugt zufällig verteilte Werte für mu und phi nach der
        Verwerfungsmethode.
    guess_kn: "Rät" Werte für die Klein-Nishina-Funktion, um so zufällige mu
        zu bestimmen. Verwendet in get_angles.
    klein_nishina: Gibt (bis auf einen konstanten Faktor) den Funktionswert der
        Klein-Nishina-Funktion zurück, zu als Input gegebenem mu und E.
    get_direction: Erzeugt aus phi und mu "lokale" Richtungsvektoren, bezogen
        auf die aktuelle Ausbreitungsrichtung jedes Teilchens. Anschließend
        wird für jedes Teilchen eine Drehmatrix K_tilde erzeugt, mit der die
        lokalen Richtungsvektoren in globale transformiert werden, sodass ein
        neuer Wert für direction erzeugt wird.
    E_scatter: Passt nach Streuung die Teilchenenergie an.
    move: Bewegt alle Teilchen um eine mittlere freie Weglänge entsprechend der
        in direction hinterlegten Richtung weiter.
    mean_free: Gibt ein Array mit zufällig verteilten mittleren freien
        Weglängen zurück.
    cleanup: Löscht Teilchen auf Basis der Mindestenergie und Mindestwichtung
        aus allen Tabellen.
    """

    def __init__(self, number=1e5, initial_energy=.1405, E_min=1e-3, W_min=1e-2):
        """
        number: Anzahl zu erzeugender Teilchen, default 1e5
        initial_energy: Anfangsenergie (MeV) der Teilchen, default 0.1405
        E_min: Energie (MeV), die Teilchen mindestens noch haben müssen um
            weiter berechnet zu werden. default 1e-3.
        W_min: Wichtung, unterhalb derer Teilchen gelöscht werden. default 1e.2
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

        self.properties = vars(self).keys()
        self.properties.remove("count")
        self.min_energy = E_min
        self.min_weight = W_min


    def interact(self,particle_mask = None):
        """
        particle_mask: Boolean Array, genau self.count Einträge. Default-Wert
            ist ein Array das alle Teilchen aktiv setzt.
        Dient als Einstiegsfunktion für die Teilcheninteraktion. Übergibt allen
        anderen Funktionen die passenden Parameter, ruft sie in der richtigen
        Reihenfolge auf und wendet sie nur auf die in particle_mask gegebenen
        Teilchen an.

        Bei Teilchen die einen Photoeffekt durchführen, wird die Wichtung
        entsprechend angepasst. Je nach Wert für das Mindestgewicht bedeu-
        tet dies sofortige Absorption oder Überleben mit neuem Gewicht.

        Ruft Funktionen in Reihenfolge
            get_angles
            get_direction
            E_scatter
            move
            cleanup
        auf.
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
        self.cleanup()

    def get_angles(self, particle_mask = None):
        """
         particle_mask: Boolean Array, genau self.count Einträge. Default-Wert
            ist ein Array das alle Teilchen aktiv setzt.
        Verwendet guess_kn() und klein_nishina() um mittels Verwerfungsmethode
        neue, zufällige Einträge für self.mu und self.phi zu generieren.
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
        count: int, Anzahl an Wertepaaren mu und f(mu), die geraten werden.

        Rät µ und klein_nishina(µ), zwecks Verwendung beim Auswürfeln von µ per
        Verwerfungsmethode. Da µ zwischen -1 und 1 liegt, der Funktionswert
        der Klein-Nishina-Formel aber zwischen 0 und 2, lassen sich beide
        auf einmal würfeln und anschließend eine Spalte um -1 verschieben.

        Gibt Numpy-Array mit den gewürfelten Werten zurück.
        """
        return 2 * np.random.rand(count,2) - np.array([1,0])

    def klein_nishina(self, mu, energy):
        """
        mu: Array mit Werten von -1 bis 1
        energy: Array mit Werten für Teilchenenergie (MeV)

        Gibt den Wert der KN-Formel zu gegebenem µ und E zurück. Vorfaktoren
        weggelassen, das wäre nur unnötige Rechnerei.
        """
        y = energy/.511
        return 1./(1 + y * (1 - mu))**2 * (1 + mu**2 + ((y**2)*(1 - mu)**2)/\
            (1 + y*(1 - mu)))

    def get_direction(self, particle_mask = None):
        """
         particle_mask: Boolean Array, genau self.count Einträge. Default-Wert
            ist ein Array das alle Teilchen aktiv setzt.

        Errechnet basierend auf den ausgewürfelten Werten für phi und mu einen
        neuen Richtungsvektor für jedes Teilchen. Hierzu wird die Drehmatrix K~
        für alle ermittelten lokalen Richtungsvektoren aufgestellt und die
        Streurichtungsvektoren damit transformiert.

        Laufzeitvariablen:
        K_tilde: Drehmatrix, (n,3,3)-Array mit n Anzahl der aktiven Teilchen.
        local_direction: Array mit über self.mu und self.phi bestimmten lokalen
            Streurichtungen.
        x,y,else_mask: Entsprechend (5)in Aufgabenstenstellung wird erkannt,
            welche Vektoren für K_tilde[:,:,1] gesetzt werden können.

        Setzt self.direction = K_tilde * local_direction (Matrixmultiplikation)
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
         particle_mask: Boolean Array, genau self.count Einträge. Default-Wert
            ist ein Array das alle Teilchen aktiv setzt.

        Aktualisiert die Teilchenenergien nach einem Stoß. Ändert self.energy
            auf den neuen Wert.
        """
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count) == 1

        self.energy[particle_mask] /= (1 + (self.energy[particle_mask]/.511) *\
            (1 - self.mu[particle_mask]))

    def move(self, particle_mask = None):
        """
        particle_mask: Boolean Array, genau self.count Einträge. Default-Wert
            ist ein Array das alle Teilchen aktiv setzt.

        Ruft self.mean_free() auf, bewegt Teilchen entsprechend dem return und
            self.direction eine zufällige Strecke.
        """
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count,dtype=bool)

        self.coords[particle_mask] += self.direction[particle_mask] * \
            self.mean_free(particle_mask)



    def mean_free(self, particle_mask = None):
        """
         particle_mask: Boolean Array, genau self.count Einträge. Default-Wert
            ist ein Array das alle Teilchen aktiv setzt.

        Spuckt eine Runde freie Weglängen aus, basierend auf den derzeitigen
        Werten für die Wirkungsquerschnittssumme.
        """
        if np.all(particle_mask) == None:
            particle_mask = np.ones(self.count,bool)

        return np.reshape(-1./self.total_x[particle_mask] * \
            np.log(np.random.rand(self.count)),(-1,1))

    def cleanup(self):
        """
        Löscht alle Teilchen die die Kriterien in self.min_energy und
            self.min_weight nicht mehr erfüllen.
        """
        cutoff = (self.energy > self.min_energy) * \
            (self.weight > self.min_weight)
        for element in self.properties:
                vars(self)[element] = vars(self)[element][cutoff]

class mc_exp(object):
    """
    Enthält das eigentliche Experiment, inklusive des Vorwissens über den
    Versuchsaufbau (Wasserkugel, Bleikollimator...). Zudem findet das Input
    Sanitizing für die anderen Klassen statt.

    Variablen:
    self.init_E: Anfangsenergie die allen Teilchen mitgegeben wird.
    self.init_count: Die anfängliche Zahl an Teilchen.
    self.new_lead: Maske, die alle Teilchen markiert die im letzten
        Iterationsschritt die Wasserkugel verlassen haben.

    Instanzen:
    self.water: interpolate-Instanz mit Wasserdaten.
    self.lead: interpolate-Instanz mit Bleidaten.
    self.particles: particles-Instanz.

    Funktionen:
    self.update_xsect: Überschreibt die Einträge für Querschnitte mit jenen
    für das jeweilig umgebende Material (Wasser innerhalb der Kugel, Blei außer-
    halb).
    """
    def __init__(self, number_of_particles=1e5, initial_energy=0.1405, E=1e-3,
            W=1e-2):
        """
        number_of_particles: Anzahl zu simulierender Teilchen, default 1e5
        initial_energy: Anfangsenergie (MeV), default 0.1405
        E: Mindestenergie für Teilchenüberleben, default 1e-3
        W: Mindestgewicht für Teilchenüberleben, default 1e-2

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

        self.particles = particles(number_of_particles,self.init_E,E,W)

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
        self.new_lead *= np.logical_not(self.water_mask)

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

    def move_particles(self):
        """
        Bewegt die Teilchen entsprechend der experimentellen Parameter weiter.
        """
        if np.any(self.water_mask) == True:
            self.water_mask = self.water_mask[(self.particles.energy >
                self.particles.min_energy) *
                (self.particles.weight > self.particles.min_weight)]
            self.update_xsect()
            self.particles.interact(self.water_mask)
        else:
            print("All particles outside of water sphere.")

    def cull_particles(self):
        """
        Killt alle Teilchen, die keine Chance haben den Kollimator zu
        erreichen (x-Komponente des Richtungs- oder Ortsvektors negativ). Nach
        einiger Überlegung bin ich der Meinung, dass Teilchen die nach
        Verlassen der Wasserkugel bei x < 0 stehen nicht mehr zum Kollimator
        gelangen können... ich kann mich irren.
        """
        survive = (self.particles.direction[:,0] > 0) *\
            (self.particles.coords[:,0] > 0)

        for element in self.particles.properties:
            vars(self.particles)[element] = vars(self.particles)[element]\
                [survive]

    def move_to_coll(self):
        """
        Setzt alle Teilchen auf die Ebene der Kollimatoroberseite, beendet
        Trajektorien die am "Detektor" vorbeilaufen.

        Ferner werden die Teilchenposition oberhalb des Kollimators, unterhalb
        des Kollimators und in der Detektorebene gespeichert.
        """
        self.particles.coords += np.reshape((200 - self.particles.coords[:,0])/
            self.particles.direction[:,0],(-1,1)) * self.particles.direction

        self.colhit_ratio = np.sum(self.particles.weight[(np.absolute(
            self.particles.coords[:,1]) < 150.25) *
            (np.absolute(self.particles.coords[:,2]) < 150.25)])/self.init_count

        self.particles.coords += np.reshape((235 - self.particles.coords[:,0])/
            self.particles.direction[:,0],(-1,1)) * self.particles.direction

        does_hit = (np.absolute(
            self.particles.coords[:,1]) < 150.25) * \
            (np.absolute(self.particles.coords[:,2]) < 150.25)

        for element in self.particles.properties:
            vars(self.particles)[element] = vars(self.particles)[element]\
                [does_hit]

        self.under_coll = self.particles.coords + np.reshape(
            (225 - self.particles.coords[:,0])/\
            self.particles.direction[:,0],(-1,1)) * self.particles.direction

        self.over_coll = self.particles.coords + np.reshape(
            (200 - self.particles.coords[:,0])/\
            self.particles.direction[:,0],(-1,1)) * self.particles.direction

        self.colpath_dir = self.under_coll - self.over_coll
        self.colpath_val = np.reshape(np.sqrt(np.sum(self.colpath_dir**2,1)),
           (-1,1))
        self.colpath_dir /= self.colpath_val

        self.particles.count = len(self.particles.coords)

    def lead_length(self, steps):
        """
        Da mir die Zeit für eine mundgemalte analytische Lösung fehlt, hier
        eine langsame, hässliche und ungenaue Methode. Es lebe die
        Diskretisierung!

        steps: Anzahl der Schritte, in die die Wegstrecke von Kollimatorober-
        zu unterseite eingeteilt wird. Läuft einen kleinen Schritt, prüft ob
        sich Teilchen in Blei befinden, läuft weiter. Hieraus berechnet sich
        ein Verhältnis Blei zu Luft, mittels dessen dann die Entscheidung fällt
        ob Teilchen absorbiert werden.
        """
        steps *= 1.
        self.current_pos = self.over_coll
        stepsize = self.colpath_val/steps
        lead_count = np.zeros(self.particles.count)
        for step in np.arange(steps):
            self.current_pos += stepsize * self.colpath_dir
            lead_count += self.is_lead()
        self.lead_ratio = np.reshape(lead_count/steps,(-1,1))
        self.lead_thickness = self.colpath_val * self.lead_ratio


    def is_lead(self):
        """
        Prüft, ob sich derzeit Teilchen innerhalb von Bleisepten aufhalten.
        Das Kollimatorraster hat eine "Wiederholrate" von 3 cm, daher wird
        mod3 verwendet um die Prüfung leichter handhabbar zu machen.
        """
        scaled_pos = np.abs(self.current_pos[:,1::]) % 3
        y = (scaled_pos[:,0] > 0.25) * (scaled_pos[:,0] < 2.75)
        z = (scaled_pos[:,1] > 0.25) * (scaled_pos[:,1] < 2.75)
        return np.logical_not(y * z)


    def poll_1(self):
        """
        Sammelt Daten für Aufgabe a) und gibt die entsprechende Prozentzahl
        aus.
        """
        self.q1 = np.round(np.sum((np.abs(
        self.particles.coords[:,1]) < self.particles.coords[:,0] * (150.25/200))
            * (np.abs(self.particles.coords[:,2]) <
            self.particles.coords[:,0]*(150.25/200)) *
            (self.particles.coords[:,0] > 0))/self.init_count*100,2)

        print("Initial in Raumwinkel emittierte Photonen: {0}%".
            format(self.q1))

    def poll_2(self):
        """
        Sammelt Daten für Aufgabe b) und gibt die entsprechende Prozentzahl
        aus.
        """
        self.q2 = np.round(np.sum(self.particles.weight)/self.init_count*100,2)
        print("Anteil an Photonen die die Wasserkugel verlassen: {0}%".
            format(self.q2))

    def poll_3(self):
        """
        Sammelt Daten für Aufgabe c) und gibt die entsprechende Prozentzahl
        aus.
        """
        self.q3 = np.round(self.colhit_ratio*100.,2)
        print("Anteil an Photonen die auf Kollimator auftreffen: {0}%".
            format(self.q3))

    def poll_4(self):
        """
        Sammelt Daten für Aufgabe d) und gibt die entsprechende Prozentzahl
        aus.
        """
        self.survivors = (self.lead_thickness < self.particles.mean_free()).\
            flatten()
        self.q4 = np.round(np.sum(self.survivors)/self.init_count*100,2)
        print("Anteil an Photonen die sowohl durch Kollimator gelangen als "\
            "auch auf Detektor auftreffen: {0}%".format(self.q4))

    def plot(self):
        """
        Gibt die räumliche Verteilung sowie die Energiespektren der Bereiche
        innerhalb eines 4 cm Radius um den Nullpunkt und außerhalb dessen
        in Konsole und Datei aus.
        """
        plt.figure()
        plt.hist2d(self.particles.coords[:,1],self.particles.coords[:,2],
                   bins=100)
        plt.colorbar()
        plt.title("Verteilung auf Detektor")
        plt.xlabel("y-Position")
        plt.ylabel("z-Position")
        plt.savefig("distribution.png")

        self.inner = (np.sqrt(np.sum(self.particles.coords[:,1::]**2,1)) < 40)\
            * self.survivors
        plt.figure()
        plt.hist(self.particles.energy[self.inner]*1e3,bins=50)
        plt.title("Spektrum in 4 cm Radius")
        plt.xlabel("E / keV")
        plt.ylabel("Anzahl")
        plt.savefig("inner.png")

        plt.figure()
        plt.hist(self.particles.energy[np.logical_not(self.inner)]*1e3,bins=50)
        plt.title("Spektrum ausserhalb")
        plt.xlabel("E / keV")
        plt.ylabel("Anzahl")
        plt.savefig("outer.png")


"""
Und los gehts, alles aufrufen und starten! Wohoooo!
"""
import mc_exp as mc

print("Beginne Simulation, erzeuge Startarrays...")
a = mc.mc_exp(1e7, W=.99)
b = a.particles
a.poll_1()

print("Beginne Bewegung in Wasser, Fortschritt\n0%")
while b.count:
    a.move_particles()
    print("{0}%".format((1 - np.sum(a.water_mask)/float(len(a.water_mask)))*
        100))

a.poll_2()
a.cull_particles()
a.move_to_coll()
a.poll_3()

a.lead_length(5e3)

a.poll_4()
a.plot()