import sys
from mpi4py import MPI
import numpy as np
from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

configfile = sys.argv[1]
config = Config(configfile)
    
n = int(sys.argv[2])    
# ================== Calculate data vectors ==========================

cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_sigma8_list(params_list, rank):
    train_params_list      = []
    train_sigma8_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        params_arr  = np.array(list(params_list[i].values()))
        sigma8 = cocoa_model.calculate_sigma8(params_list[i])
        train_params_list.append(params_arr)
        train_sigma8_list.append(sigma8)
    return train_params_list, train_sigma8_list

def get_sigma8s(params_list, comm, rank):
    local_params_list, local_sigma8_list = get_local_sigma8_list(params_list, rank)
    if rank!=0:
        comm.send([local_params_list, local_sigma8_list], dest=0)
        train_params       = None
        train_sigma8s = None
    else:
        sigma8_list = local_sigma8_list
        params_list      = local_params_list
        for source in range(1,size):
            new_params_list, new_sigma8_list = comm.recv(source=source)
            sigma8_list = sigma8_list + new_sigma8_list
            params_list      = params_list + new_params_list
        train_params       = np.vstack(params_list)    
        train_sigma8s = np.vstack(sigma8_list)        
    return train_params, train_sigma8s


print("Iteration: %d"%(n))

next_training_samples = np.load(config.savedir + '/' + config.chainname + '_%d.npy'%(n))
rng = np.random.default_rng(0)
idx = rng.choice(next_training_samples.shape[0], size=8080, replace=False)

n_params = len(config.param_labels)
subsamples = next_training_samples[idx]
params_list = get_params_list(subsamples[:, :n_params], config.param_labels)
if config.probe == 'cosmic_shear':
    n_extra = subsamples.shape[1] - n_params
    for k, item in enumerate(params_list):
        for i in range(n_extra):
            item[f'extra_{i}'] = subsamples[k, n_params + i]

current_iter_samples, current_iter_sigma8s = get_sigma8s(params_list, comm, rank)

train_sigma8s = current_iter_sigma8s

# ================== Train emulator ==========================
if(rank==0):
    train_sigma8s = np.concatenate(train_sigma8s)
    new_samples = np.column_stack((current_iter_samples, train_sigma8s))
    # ========================================================
    np.save(config.savedir + '/new_' + config.chainname + '_%d.npy'%(n), new_samples)
    # ======================================================== 

MPI.Finalize
