import argparse
import numpy as np
import torch

from cocoa_emu import Config
from cocoa_emu.emulator import NNEmulator
import emu_tools

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('-N', type=int)

args = parser.parse_args()

configfile = args.config
n = int(args.N)


#==============================================

config = Config(configfile)

#==============================================

train_samples      = np.load(config.emudir + '/train_samples_%d.npy'%(n))
train_data_vectors = np.load(config.emudir + '/train_data_vectors_%d.npy'%(n))

#=================================================

def get_chi_sq_cut(train_data_vectors):
    chi_sq_list = []
    for dv in train_data_vectors:
        delta_dv = (dv - config.dv_obs)[config.mask]
        chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
        chi_sq_list.append(chi_sq)
    chi_sq_arr = np.array(chi_sq_list)
    select_chi_sq = (chi_sq_arr < config.chi_sq_cut)
    return select_chi_sq

select_chi_sq = get_chi_sq_cut(train_data_vectors)
selected_obj = np.sum(select_chi_sq)
total_obj    = len(select_chi_sq)
print("Total number of objects: %d"%(selected_obj))

train_data_vectors = train_data_vectors[select_chi_sq]
train_samples      = train_samples[select_chi_sq]
#==============================================
torch.set_num_threads(48)
    
full_emu = NNEmulator(config.n_dim, config.output_dims, config.dv_fid, config.dv_std, config.mask_ones, config.nn_model)
#==============================================
N_Z_BINS = 5
N_angular_bins = 26

ggl_exclude = []

N_xi  = emu_tools.N_xi
N_ggl = emu_tools.N_ggl
N_w   = emu_tools.N_w

dv_ggl = np.zeros(N_ggl)
dv_w   = np.zeros(N_w)

print("N_xi: %d"%(N_xi))

emu_xi_plus = emu_tools.get_NN_emulator('xi_plus', config)
emu_xi_minus = emu_tools.get_NN_emulator('xi_minus', config)

print("=======================================")
print("Training xi_plus emulator....")
emu_xi_plus.train(torch.Tensor(train_samples), torch.Tensor(train_data_vectors[:,:N_xi]),\
            batch_size=config.batch_size, n_epochs=config.n_epochs)
print("=======================================")
print("=======================================")
print("Training xi_minus emulator....")
emu_xi_minus.train(torch.Tensor(train_samples), torch.Tensor(train_data_vectors[:,N_xi:2*N_xi]),\
            batch_size=config.batch_size, n_epochs=config.n_epochs)
print("=======================================")


emu_xi_plus.save(config.emudir + '/xi_p_%d'%(n))
emu_xi_minus.save(config.emudir + '/xi_m_%d'%(n))
