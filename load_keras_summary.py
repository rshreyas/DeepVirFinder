import os, sys, optparse, warnings
import h5py, multiprocessing
import numpy as np

import theano
import keras
from keras.models import load_model

modDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
modPattern = 'model_siamese_varlen_0.3k'
modName = [ x for x in os.listdir(modDir) if modPattern in x and x.endswith(".h5") ][0]

print(modName)

deepvirfinder_model = load_model(os.path.join(modDir, modName))

print(deepvirfinder_model.summary())
