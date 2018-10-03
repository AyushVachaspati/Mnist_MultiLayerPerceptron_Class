import tensorflow as tf
import numpy as np
import Mnist
from Preprocessing import get_datset

train_pixels,train_labels,test_pixels,test_labels = get_dataset()

model = Mnist(self,num_pixels=28*28,batch_size=64,num_labels=10,layers_cells=[800,100])
model.build_graph()
model.train(epoch=100,train_pixels=train_pixels,
	   train_labels=train_labels,test_pixels=test_pixels,
	   test_labels=test_labels)
