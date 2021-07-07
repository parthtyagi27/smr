import numpy
import numpy as np
import tensorflow as tf
import h5py
import gwsurrogate
from mpi4py import MPI
from time import time

class Generator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, batch_size=16, dim=(101300,), n_channels=1,
				 shuffle=True, hvd_rank=0, hvd_size=1):
		'Initialization'
		self.sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
		self.li = self.__generate_indices(step_q=8*0.01, step_chi=0.012)
		#self.chunk = 16  # Number of waveforms to generate on each rank at each iter [Limited by memory on the node]
		self.chunk = batch_size
		#self.file_path = file_path
		#self.f = h5py.File(self.file_path, 'r')
		#self.keys = list(self.f.keys())
		#self.dset = self.f[self.keys[0]]
		#self.lset = self.f[self.keys[1]]
		
		self.dim = dim
		self.batch_size = batch_size
		self.n_channels = n_channels
		self.shuffle = shuffle
		self.rank = hvd_rank
		self.size = hvd_size
		self.epoch = 0
		print("last function")
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.li) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		#pos = self.local_indices[index*self.batch_size:(index+1)*self.batch_size]
		#if self.shuffle == True: #Sorting is necessary to read h5 files
		#	 pos = np.sort(pos)

		# Generate data
		#X, y = self.__data_generation(pos)
		X, y = self.__data_generation(index)
		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		#self.global_indices = np.arange(len(self.dset))
		#if self.shuffle == True:
		#	np.random.seed(self.epoch); np.random.shuffle(self.global_indices)
		#
		#chunk_size = len(self.global_indices) // self.hvd_size
		#self.local_indices = self.global_indices[self.hvd_rank*chunk_size: (self.hvd_rank + 1)*chunk_size]
		self.epoch += 1

	def __generate_waveform(self, params):
		q = params[0]
		chiA = [0, 0, params[1]]
		chiB = [0, 0, params[2]]
		f_low = 0
		times = np.arange(-10000,130,0.1) # The module only allows a maximum of 130 M after the event
		times, h, dyn = self.sur(q, chiA, chiB, times=times, f_low=f_low)

		return h[(2,2)].real
   

	def __generate_indices(self, step_q, step_chi):
		li = []
		for q in np.arange(1,8,step_q):
			for chiA in np.arange(-0.8,0.8,step_chi):
				for chiB in np.arange(-0.8,0.8,step_chi):
					li.append((q,chiA,chiB))
		return np.asarray(li)
 
	def __data_generation(self, index):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		#labels = self.li[index : index + self.chunk * self.size]
		print("Epoch = ", self.epoch, "Index = ", index)
		labels = self.li[index : index + self.batch_size]			
		#if self.rank == self.size - 1:
		#	batch = labels[self.rank * self.chunk:]
		#else:
		#	batch = labels[self.rank * self.chunk: (self.rank+1)*self.chunk]
		batch = labels
		print("batch shape " + str(batch.shape))
		print(self.size)	
		start_time = time()
		#waveforms = generate_batch_of_waveforms(batch)
		waveforms = list(map(self.__generate_waveform, batch))
		duration =	time() - start_time
   
		#comm = MPI.COMM_WORLD
		#print("waveform len = ", len(waveforms))
		#print(self.rank , " ", waveforms)
		# Gather/reduce all the waveforms generated at all ranks in this iter
		#G = comm.reduce(waveforms, op=MPI.SUM, root=0)
		G = waveforms
		#print("g len = ", len(G))
		#print(G)
		if self.rank == 0:
			print('len waveforms : ', len(G))
			print('len labels: ', len(labels))
	
			# On rank 0 save the gathered waveforms and labels at the appropriate loc in hdf5 file
			#f = h5py.File("train.hdf5", "r+")
			#dset = f["/data/"]
			#lset = f["/labels/"]

			#dset[i: i+chunk*size] = G
			#lset[i: i+chunk*size] = labels

			#f.close()
		
			

			#print('Iter: ', i)
			print('My rank is ', self.rank, '/', self.size)
			print('Length of Batch on rank: ', self.rank, ' is ', len(batch))
			print('Shape of Result on rank: ', self.rank, ' is ', len(waveforms), type(waveforms))
			print( "Generated %s waveforms in %s seconds" %(len(waveforms), duration) )
			print(' ')
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size), dtype=int)
		
		#X[:,:,0] = self.dset[indexes, :]
		#y = self.lset[indexes, :]
	   
		X[:, :, 0] = G
		y = labels
 
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
		print(label.shape)
		print(y.shape)
		label[:, 0] = y[:, 0]
		label[:, 1] = y[:, 1]
		label[:, 2] = y[:, 2]
		label[:, 3] = chi
		label[:, 4] = S_eff
		#label[:, 5] = Sigma
		
		return X, [label[:,0], label[:, 1:]]
