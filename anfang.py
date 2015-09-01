import numpy as np

class collimator(object):
	"""
	Beschreibt einen quadratischen Bleikollimator. Material und Form sind
	also fest, lediglich die Außenmaße und die Lochabmessungen können
	vorgegeben werden.
	"""
	def __init__(self, width, height, holesize):
		super(collimator, self).__init__()
		self.width = width
		self.height = height
		self.holesize = holesize
		self.

	def mean_free(self, energy):
		"""
		Muss noch insoweit angepasst werden, dass die korrekten Querschnitte
		für die entsprechenden Energien verwendet werden.
		"""
		return -1/self.total_ia * np.log(np.random.uniform(size=shape))