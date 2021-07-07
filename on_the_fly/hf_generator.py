import numpy
import numpy as np
import tensorflow as tf
import h5py



class h5Generator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_path, batch_size=32, dim=(101300,), n_channels=1,
                 shuffle=True, hvd_rank=0, hvd_size=1):
        'Initialization'
        self.file_path = file_path
        self.f = h5py.File(self.file_path, 'r')
        self.keys = list(self.f.keys())
        self.dset = self.f[self.keys[0]]
        self.lset = self.f[self.keys[1]]
        
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.hvd_rank = hvd_rank
        self.hvd_size = hvd_size
        self.epoch = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.local_indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        pos = self.local_indices[index*self.batch_size:(index+1)*self.batch_size]
        if self.shuffle == True: #Sorting is necessary to read h5 files
            pos = np.sort(pos)

        # Generate data
        X, y = self.__data_generation(pos)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.global_indices = np.arange(len(self.dset))
        if self.shuffle == True:
            np.random.seed(self.epoch); np.random.shuffle(self.global_indices)
        
        chunk_size = len(self.global_indices) // self.hvd_size
        self.local_indices = self.global_indices[self.hvd_rank*chunk_size: (self.hvd_rank + 1)*chunk_size]
        self.epoch += 1

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        X[:,:,0] = self.dset[indexes, :]
        y = self.lset[indexes, :]
        
        # Last shuffle so that mass ratios aren't fed in ascending order
        if self.shuffle == True:
            assert len(X) == len(y)
            p = numpy.random.permutation(len(X))
            X, y = X[p], y[p]
	    
        
        #chi = (y[:, 0]*y[:, 1] + y[:, 2]) / (y[:, 0] + 1)
        
        
        #q, s1, s2 = y[:, 0], y[:, 1], y[:, 2]
        #chi = ( q*s1 + s2 )/( q + 1 )
        #sigma_1 = 1 + 3/(4*q)
        #sigma_2 = 1 + (3*q)/4
        #S_eff = sigma_1*s1 + sigma_2*s2
        #Sigma = -(1/4)*(s1/q + q*s2)
        
        chi = ( y[:,0]*y[:,1] + y[:,2] )/( y[:,0] + 1 )
        sigma_1 = 1 + 3/(4*y[:,0])
        sigma_2 = 1 + (3*y[:,0])/4
        S_eff = sigma_1*y[:,1] + sigma_2*y[:,2]
        Sigma = -(1/4)*(y[:,1]/y[:,0] + y[:,0]*y[:,2])


        label = np.empty((self.batch_size, 5))
        label[:, 0] = y[:, 0]
        label[:, 1] = y[:, 1]
        label[:, 2] = y[:, 2]
        label[:, 3] = chi
        label[:, 4] = S_eff
        #label[:, 5] = Sigma
        
        return X, [label[:,0], label[:, 1:]]
