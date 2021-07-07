import numpy
import numpy as np
import tensorflow as tf
#import gwsurrogate
#import generate as generate

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K




# Create Model
def create_model(last_n_steps=10000):
    # convolutional operation parameters
    n_filters = 32 
    filter_width = 2
    dilation_rates = [2**i for i in range(8)] * 2 

    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    Input_seq = Input(shape=(101300, 1), dtype='float32')
    x = Input_seq

    skips = []
    for dilation_rate in dilation_rates:

        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(16, 1, padding='same')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)	

        # filter convolution
        x_f = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='same',
                     dilation_rate=dilation_rate)(x)
        x_f = BatchNormalization()(x_f)  

        # gating convolution
        x_g = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='same',
                     dilation_rate=dilation_rate)(x)
        x_g = BatchNormalization()(x_g)

        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])

        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(16, 1, padding='same')(z)
        z = BatchNormalization()(z)	
        z = Activation('relu')(z)	

        # residual connection
        x = Add()([x, z])    

        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers 
    out = Conv1D(128, 1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(.2)(out)
    out = Conv1D(1, 1, padding='same')(out)
    out = BatchNormalization()(out)

    # extract the last 60 time steps as the training target
    def slice(x, seq_length):
        return x[:,-seq_length:,:]

    pred_seq = Lambda(slice, arguments={'seq_length':last_n_steps})(out)

    # extract mass_ratio and spins
    out = Flatten()(pred_seq)
    
    # q branch
    out_1 = Dense(512, activation='relu')(out)
    #out_1 = Dropout(0.2)(out_1)
    pred_q = Dense(1, name='q_output')(out_1)

    # s1,s2,chi,sigma branch
    out_2 = Dense(512, activation='relu')(out)
    #out_2 = Dropout(0.2)(out_2)
    pred_ap = Dense(4, name='ap_output')(out_2)

    model = Model(Input_seq, [pred_q, pred_ap])
    
    return model




# Define Data Loader
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, IDs, batch_size=32, dim=(101300,), n_channels=1,
                 shuffle=True, hvd_rank=0, hvd_size=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.IDs = IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.hvd_rank = hvd_rank
        self.hvd_size = hvd_size
        self.epoch = 0
        self.on_epoch_end()
        self.sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8') 

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        IDs = np.copy(self.IDs)
        if self.shuffle == True:
            np.random.seed(self.epoch); np.random.shuffle(IDs)
        chunk_size = len(IDs) // self.hvd_size
        #if self.hvd_rank == (self.hvd_size - 1):
        #    self.list_IDs = IDs[self.hvd_rank*chunk_size: ]
        #else:
        #    self.list_IDs = IDs[self.hvd_rank*chunk_size: (self.hvd_rank + 1)*chunk_size]
        self.list_IDs = IDs[self.hvd_rank*chunk_size: (self.hvd_rank + 1)*chunk_size]
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.epoch += 1
            
    def generate_waveform(self, params):
        q = params[0]
        chiA = [0, 0, params[1]]
        chiB = [0, 0, params[2]]
        f_low = 0
        times = np.arange(-10000,130,0.1) # The module only allows a maximum of 130 M after the event
        times, h, dyn = self.sur(q, chiA, chiB, times=times, f_low=f_low)

        return h[(2,2)].real
        

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        for i, ID in enumerate(list_IDs_temp):
            X[i,:,0] = self.generate_waveform(ID)
        
        y = np.stack( list_IDs_temp, axis=0 )
        y[:,0] = 1/y[:,0]

        return X, y   
    

    
    
# Define custom CallBacks
class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes batch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(batch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(batch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (batch + 1, lr))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
                
            
class LossHistory(tf.keras.callbacks.Callback):
    
    def __init__(self, filename, delimiter=','):
        self.filename = filename
        self.delimiter = delimiter
        
        
    def on_epoch_begin(self, epoch, logs={}):
        self.losses = []
        self.f = self.filename.split('.')[0]+'_epoch_%s.'%(epoch+1)+ self.filename.split('.')[-1]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        numpy.savetxt(self.f, numpy.array(self.losses), delimiter=self.delimiter)
           



            
def create_generator(step_q=10, step_chi=10, batch_size=4, shuffle=False, 
                     store_ind=True, load_ind=False, filepath='', rank=0, size=1):
      
    def generate_indices(step_q,step_chi):
        li = []
        for q in np.arange(0.125,1,step_q):
            for chiA in np.arange(-0.8,0.8,step_chi):
                for chiB in np.arange(-0.8,0.8,step_chi):
                    li.append(((1/q),chiA,chiB))
        return np.asarray(li)

    def store_indices(indices,file_name):
        np.save(file_name, indices)

    def load_indices(file_name):
        return np.load(file_name)
    
    
    if load_ind==1:
        li = load_indices(filepath)
        generator = DataGenerator(li, batch_size=batch_size, shuffle=shuffle, hvd_rank=rank, hvd_size=size)
        
    else:
        li = generate_indices(step_q, step_chi)
        generator = DataGenerator(li, batch_size=batch_size, shuffle=shuffle, hvd_rank=rank, hvd_size=size)
        if store_ind==1:
            store_indices(li, filepath)
    
    return generator
