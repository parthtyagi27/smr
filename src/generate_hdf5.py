import h5py
import numpy as np
import gwsurrogate
from mpi4py import MPI
from time import time
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(size, " - ", rank)

# Generates a single waveform using the given params and returns the
def generate_waveform(params):
    q = params[0]
    chiA = [0, 0, params[1]]
    chiB = [0, 0, params[2]]
    f_low = 0
    times = np.arange(-10000,130,0.1) # The module only allows a maximum of 130 M after the event
    times, h, dyn = sur(q, chiA, chiB, times=times, f_low=f_low)

    return h[(2,2)].real

# Generate labels
def generate_indices(step_q,step_chi):
        li = []
        for q in np.arange(1,8,step_q):
            for chiA in np.arange(-0.8,0.8,step_chi):
                for chiB in np.arange(-0.8,0.8,step_chi):
                    li.append((q,chiA,chiB))
        return np.asarray(li)






if __name__ == "__main__":
    
    sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
   
    li = generate_indices(step_q=8*0.01, step_chi=0.012)
    chunk = 16  # Number of waveforms to generate on each rank at each iter [Limited by memory on the node]

    # Create hdf5 file on rank 0
    if rank == 0:
        f = h5py.File("train.hdf5", "w")
        dset = f.create_dataset("data", (len(li), 101300), dtype='float64')
        lset = f.create_dataset("labels", (len(li),3), dtype='float64')
        f.close()
    
    
    print(len(li), " step = ", chunk*size)
    print(size)   
    # Go through all labels in steps of chunk*size [Each rank will generate chunk of waveforms except for may be last iter]
    for i in range(0, len(li), chunk*size):
        labels = li[i: i+chunk*size]

        print("i = ", i, " len = ", chunk * size , " : ", labels.shape) 
 
        if rank == size - 1:
            batch = labels[rank*chunk:]
        else:
            batch = labels[rank*chunk: (rank+1)*chunk]
        
        start_time = time()
        #waveforms = generate_batch_of_waveforms(batch)
        waveforms = list(map(generate_waveform, batch))
        duration =  time() - start_time
   

        # Gather/reduce all the waveforms generated at all ranks in this iter
        G = comm.reduce(waveforms, op=MPI.SUM, root=0)
        
        if rank == 0:
            print('len waveforms : ', len(G))
            print('len labels: ', len(labels))
    
            # On rank 0 save the gathered waveforms and labels at the appropriate loc in hdf5 file
            f = h5py.File("train.hdf5", "r+")
            dset = f["/data/"]
            lset = f["/labels/"]

            dset[i: i+chunk*size] = G
            lset[i: i+chunk*size] = labels

            f.close()

            print('Iter: ', i)
            print('My rank is ',rank, '/', size)
            print('Length of Batch on rank: ', rank, ' is ', len(batch))
            print('Shape of Result on rank: ', rank, ' is ', len(waveforms), type(waveforms))
            print( "Generated %s waveforms in %s seconds" %(len(waveforms), duration) )
            print(' ')
