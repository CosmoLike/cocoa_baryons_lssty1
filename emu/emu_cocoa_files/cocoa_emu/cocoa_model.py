import numpy as np

from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.model import Model
from cobaya.conventions import kinds

_timing = "timing"
_params = "params"
_prior = "prior"
_packages_path = "packages_path"

## Hack
from collections import namedtuple
ComponentKinds = namedtuple('ComponentKinds', kinds)
kinds = ComponentKinds(*ComponentKinds._fields)


def get_model(yaml_file):
    info  = yaml_load_file(yaml_file)
    updated_info = update_info(info)
    model =  Model(updated_info[_params], updated_info[kinds.likelihood],
               updated_info.get(_prior), updated_info.get(kinds.theory),
               packages_path=info.get(_packages_path), timing=updated_info.get(_timing),
               allow_renames=False, stop_at_error=info.get("stop_at_error", False))
    return model

class CocoaModel:
    def __init__(self, configfile, likelihood):
        self.model      = get_model(configfile)
        self.likelihood = likelihood
        
    def calculate_data_vector(self, params_values, baryon_scenario=None, return_sigma8=False):        
        likelihood   = self.model.likelihood[self.likelihood]
        input_params = self.model.parameterization.to_input(params_values)
        self.model.provider.set_current_input_params(input_params)
        for (component, index), param_dep in zip(self.model._component_order.items(), 
                                                 self.model._params_of_dependencies):
            depend_list = [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(params, want_derived=False,
                                         dependency_params=depend_list, cached=False)
        if baryon_scenario is None:
            data_vector = likelihood.get_datavector(**input_params)
        else:
            data_vector = likelihood.compute_barion_datavector_masked_reduced_dim(baryon_scenario, **input_params)
        
        if return_sigma8:
                provider = self.model.provider
                sigma8 = provider.requirement_providers['omegam'].current_state['derived_extra']['sigma8']
                return np.array(data_vector), sigma8

        return np.array(data_vector)

    def calculate_sigma8(self, params_values):        
        camb_params = self.model.parameterization.to_input(params_values)
        likelihood   = self.model.likelihood[self.likelihood]
        this_camb = self.model.theory['camb'].camb
        As = camb_params['As']
        ns =camb_params['ns']
        H0 = camb_params['H0']
        omegabh2 = camb_params['omegabh2']
        omegach2 = camb_params['omegach2']
        mnu = camb_params['mnu']
        tau = camb_params['tau']
        w0 = camb_params['w']
        wa = camb_params['wa']

        params = this_camb.set_params(As=As, ns=ns, H0=H0, ombh2=omegabh2, omch2=omegach2, omk=0., mnu=mnu, nnu=3.046, 
                                      w=w0, wa=wa, tau=tau, WantTransfer=True, dark_energy_model='ppf', accurate_massive_neutrino_transfers=False, num_massive_neutrinos=1,
                                      AccuracyBoost=1.1)
        results = this_camb.get_results(params)
        sigma8 = results.get_sigma8_0()

        return np.array(sigma8)
