from tensorflow.keras.models import load_model
#import keras.backend as K
#import tensorflow as tf
import os
#from tensorflow import Session
from coloncancer import app
class coloncancer_Model():

	graph, model = None, None
	#session1 = None
	def __init__(self):
		#K.clear_session()
		# self.session1 = Session()
		# self.graph = tf.get_default_graph()
		# with self.graph.as_default():
		# 	with self.session1.as_default():
		model_path = os.path.join(app.root_path, 'static/saved_models', 'weights.hdf5')
		self.model = load_model(model_path)
		return None	

	def predict(self, img):
		# with self.graph.as_default():
		# 	with self.session1.as_default():
		return self.model.predict(img)