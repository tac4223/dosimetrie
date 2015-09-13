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

    def __init__(self, data, headersize=3):
        """
        Schlichte Initialisierung. Headersize gibt die Anzahl an Kopfzeilen an,
        die übersprungen werden müssen um zum Beginn der Zahlenwerte zu kommen.
        """
        self.data = np.loadtxt(data,skiprows=headersize)
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