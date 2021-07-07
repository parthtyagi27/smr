#!/usr/bin/env python
from __future__ import print_function
import numpy
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from training_utils import create_model, DataGenerator, LearningRateScheduler, LossHistory
from hf_generator import h5Generator



import argparse
parser = argparse.ArgumentParser(description="smr bbm")
parser.add_argument('--batch_size',type=int, help='batch size', default=64)
parser.add_argument('--checkpoint_dir', help='root directory', default='/home/khan74/checkpoint/')
parser.add_argument('--data_dir', help='root directory', default='/home/khan74/data/')
args = parser.parse_args()
batch_size = args.batch_size
checkpoint_dir = args.checkpoint_dir
data_dir = args.data_dir






import horovod.tensorflow.keras as hvd
hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

K.set_session(tf.Session(config=config))



training_generator = h5Generator(file_path= data_dir+'/train.hdf5', batch_size=batch_size, shuffle=True,
                                 hvd_rank=hvd.rank(), hvd_size=hvd.size())
validation_generator = h5Generator(file_path= data_dir+'/val.hdf5', batch_size=batch_size, shuffle=False,
                                 hvd_rank=hvd.rank(), hvd_size=hvd.size())
test_generator = h5Generator(file_path= data_dir+'/test.hdf5', batch_size=batch_size, shuffle=False,
                                 hvd_rank=hvd.rank(), hvd_size=hvd.size())





opt = Adam(5e-4)
opt = hvd.DistributedOptimizer(opt)
# model = create_model(last_n_steps=5000)
model = keras.models.load_model('/home/khan74/checkpoint/weights/best_model_19-0.00.h5')
model.compile(opt, loss='mean_squared_error')



# Horovod: write logs on worker 0.
verbose = 2 if hvd.rank() == 0 else 0



'''
t2= time()
#train_score = hvd.allreduce(model.evaluate_generator(generator=training_generator, verbose=verbose, max_queue_size=1000, workers=32, use_multiprocessing=True))
val_score = hvd.allreduce(model.evaluate_generator(generator=validation_generator, verbose=verbose, max_queue_size=1000, workers=32, use_multiprocessing=True))
test_score = hvd.allreduce(model.evaluate_generator(generator=test_generator, verbose=verbose, max_queue_size=1000, workers=32, use_multiprocessing=True))
t3 = time()


if (hvd.rank()==0):
    #print("All Reduce: train loss: %s  -  val loss: %s  -  test loss: %s" %(train_score, val_score, test_score))
    print("All Reduce: val loss: %s  -  test loss: %s" %(val_score, test_score))
    print('Inference time: %s' %(t3 - t2))
'''        
        
        

# Make predictions on each ranks chunk
p_q, p_ap = model.predict_generator(test_generator, verbose=1, max_queue_size=1000, workers=32, use_multiprocessing=False)
predictions = np.hstack([p_q, p_ap])
v_q, v_ap =  model.predict_generator(validation_generator, verbose=1, max_queue_size=1000, workers=32, use_multiprocessing=False)
validations = np.hstack([v_q, v_ap])


# Gathering all Predictions
Predictions = hvd.allgather(predictions)
np.save(checkpoint_dir+'Preds', Predictions)

Validations = hvd.allgather(validations)
np.save(checkpoint_dir + 'Vals', Validations)

# Gather labels on each ranks chunk
labels = []
for i, (_,y) in enumerate(test_generator):
    labels.append(np.hstack((y[0].reshape([-1,1]), y[1])))
    if i == len(test_generator.local_indices) - 1 :
        break
labels = np.squeeze(np.array(labels), axis=1)

# Gathering all Labels
Labels = hvd.allgather(labels)
np.save(checkpoint_dir+'Test_Labels', Labels)


# Gather labels on each ranks chunk
val_labels = []
for i, (_,y) in enumerate(validation_generator):
    val_labels.append(np.hstack((y[0].reshape([-1,1]), y[1])))
    if i == len(validation_generator.local_indices) - 1 :
        break
val_labels = np.squeeze(np.array(val_labels), axis=1)

# Gathering all Labels
Val_Labels = hvd.allgather(val_labels)
np.save(checkpoint_dir+'Val_Labels', Val_Labels)



print('hvd rank: ', hvd.rank())
print('test gen len: ', len(test_generator.global_indices))
print('This rank test gen len: ', len(test_generator.local_indices))
print('len(preds): ', len(predictions))
print('len(labels): ', len(labels))
print()
print('len(Gathered Preds): ', len(Predictions))
print('len(Gathered Labels): ', len(Labels))
print()
print('val gen len: ', len(validation_generator.global_indices))
print('len(Gathered Vals): ', len(Validations))
print('len(Gathered Labels): ', len(Val_Labels))
print()
