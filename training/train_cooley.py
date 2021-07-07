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
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from training_utils import create_model, DataGenerator, LearningRateScheduler, LossHistory
from hf_generator import h5Generator
from os import path, makedirs


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


if hvd.rank() == 0:
    if not path.exists(checkpoint_dir):
        makedirs(checkpoint_dir)
        makedirs(checkpoint_dir + '/weights/')
        makedirs(checkpoint_dir + '/logs/')




training_generator = h5Generator(file_path= data_dir+'/train.hdf5', batch_size=batch_size, shuffle=True,
                                 hvd_rank=hvd.rank(), hvd_size=hvd.size())
validation_generator = h5Generator(file_path= data_dir+'/val.hdf5', batch_size=batch_size, shuffle=True,
                                 hvd_rank=hvd.rank(), hvd_size=hvd.size())
test_generator = h5Generator(file_path= data_dir+'/test.hdf5', batch_size=1, shuffle=False,
                                 hvd_rank=hvd.rank(), hvd_size=hvd.size())





opt = Adam(5e-4)
opt = hvd.DistributedOptimizer(opt)
model = create_model(last_n_steps=10000)
#model = keras.models.load_model('/lus/theta-fs0/projects/mmaADSP/khan/SMR_BBM/checkpoint_ap_branch_w_NoDropOut_run8/weights/model_05-0.02.h5')
model.compile(opt, loss={'q_output': 'mean_squared_error', 'ap_output': 'mean_squared_error'})




batch_history = LossHistory(filename=checkpoint_dir + '/logs/batch_loss_history_rank_%s.log' %hvd.rank())
epoch_history = keras.callbacks.CSVLogger(checkpoint_dir + '/logs/epoch_loss_history.log')

Best_ModelCheckpoint = keras.callbacks.ModelCheckpoint(checkpoint_dir + '/weights/best_model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
Regular_ModelCheckpoint = keras.callbacks.ModelCheckpoint(checkpoint_dir + '/weights/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_weights_only=False, period=1)

Early_Stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='min')
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-10, verbose=1, mode='min')



callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),

    Early_Stopping,
    ReduceLROnPlateau,
    #batch_history,
]

if hvd.rank() == 0:
    callbacks.append(epoch_history)
    callbacks.append(Best_ModelCheckpoint)
    callbacks.append(Regular_ModelCheckpoint)

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0


print('hvd rank: ', hvd.rank())
print('hvd size: ', hvd.size())
print('batch_size: ', batch_size)
print('train gen len: ', len(training_generator.global_indices))
print('This rank train gen len: ', len(training_generator.local_indices))
print('steps: ', (len(training_generator.local_indices) // training_generator.batch_size))
print('val gen len: ', len(validation_generator.global_indices))
print('This rank val gen len: ', len(validation_generator.local_indices))
print('val steps: ', (len(validation_generator.local_indices) // validation_generator.batch_size))
print()
t0 = time()
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              epochs=100,
                              callbacks=callbacks,
                              max_queue_size = 1000,
                              workers = 32,
                              use_multiprocessing = False,
                              verbose=verbose)
t1 = time()

if (hvd.rank()==0):
    print('**Evaluation time: %s' %(t1-t0))
