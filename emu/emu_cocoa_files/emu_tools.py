from cocoa_emu.emulator import NNEmulator

N_Z_BINS = 5
N_angular_bins = 26

ggl_exclude = []

N_xi  = int((N_Z_BINS * (N_Z_BINS + 1)) // 2 * N_angular_bins)
N_ggl = int((N_Z_BINS * N_Z_BINS - len(ggl_exclude)) * N_angular_bins)
N_w   = int(N_Z_BINS * N_angular_bins)

def get_NN_emulator(probe, config):
    if probe=='xi_plus':
        return  NNEmulator(config.n_dim, N_xi, config.dv_fid[:N_xi], config.dv_std[:N_xi], config.mask[:N_xi], config.nn_model)
    
    elif probe=='xi_minus':
        return NNEmulator(config.n_dim, N_xi, config.dv_fid[N_xi:2*N_xi], config.dv_std[N_xi:2*N_xi], config.mask[N_xi:2*N_xi], config.nn_model)
    
    elif probe=='ggl':
        return NNEmulator(config.n_dim, N_ggl, config.dv_fid[2*N_xi:2*N_xi+N_ggl], config.dv_std[2*N_xi:2*N_xi+N_ggl], config.mask[2*N_xi:2*N_xi+N_ggl], config.nn_model)
    
    elif probe=='w':
        assert config.dv_fid[2*N_xi+N_ggl:].shape[0] == N_w
        return NNEmulator(config.n_dim, N_w, config.dv_fid[2*N_xi+N_ggl:], config.dv_std[2*N_xi+N_ggl:], config.mask[2*N_xi+N_ggl:], config.nn_model)