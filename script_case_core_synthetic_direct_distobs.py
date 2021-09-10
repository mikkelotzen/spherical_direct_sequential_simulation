from sdss import SDSS
import mikkel_tools.utility as mt_util

import numpy as np
from math import inf
from scipy.optimize import curve_fit
import scipy as sp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time
import pyshtools
import pickle

comment = "Synthetic core with tapered ensemble as prior"
nb_name = "nb_case_core_synthetic_direct_distobs"

shc_g = 30
shc_grid = 30

core = SDSS(comment, N_SH = shc_g, sim_type = "core_ens", sat_height = 350)

core.grid_glq(nmax = shc_grid, r_at = core.r_cmb)

grid_in = np.array([core.grid_phi, 90-core.grid_theta]).T
core.make_grid(core.r_cmb, grid_in, calc_sph_d = True)
core.generate_map()


s_source = SDSS(comment, N_SH = shc_g, sim_type = "core", sat_height = 350, N_SH_secondary = None)

s_source.grid_glq(nmax = shc_grid, r_at = core.r_cmb)

grid_in = np.array([s_source.grid_phi, 90-s_source.grid_theta]).T
s_source.make_grid(s_source.r_cmb, grid_in, calc_sph_d = False)

s_source.generate_map()


# Load core object
file_pickle = open("nb_case_core_synthetic_direct.obj", 'rb')
core_dobs = pickle.load(file_pickle)
file_pickle.close()


C_e_const = 2.0
#observations_direct_loc = np.random.choice(np.arange(s_source.grid_N),size=512,replace=False)
observations_direct_loc = core_dobs.observations_direct_loc
observations_direct = s_source.data[observations_direct_loc] + np.random.normal(loc=0.0,scale=C_e_const,size=observations_direct_loc.shape)

core_dobs = None

core.observations_direct_loc = observations_direct_loc


shc_g = 30
shc_grid = 30

core_dobs = SDSS(comment, N_SH = shc_g, sim_type = "core_ens", sat_height = 350)

core_dobs.grid_glq(nmax = shc_grid, r_at = core_dobs.r_cmb)

core_dobs.grid_phi = core_dobs.grid_phi[observations_direct_loc]
core_dobs.grid_theta = core_dobs.grid_theta[observations_direct_loc]

grid_in = np.array([core_dobs.grid_phi, 90-core_dobs.grid_theta]).T
core_dobs.make_grid(core_dobs.r_cmb, grid_in, calc_sph_d = True)

core_dobs.data = observations_direct
core_dobs.target_var = np.var(core_dobs.data)


core_dobs.semivar(model_lags = 51, model = "exponential", max_dist = 15000, lag_length = 100,
             zero_nugget = True, hit_target_var = False)



core.model = core_dobs.model
core.a_sv = core_dobs.a_sv
core.C0 = core_dobs.C0
core.C1 = core_dobs.C1
core.C2 = core_dobs.C2
core.C3 = core_dobs.C3

core.n_lags = core_dobs.n_lags
core.lags = core_dobs.lags
core.pics = core_dobs.pics
core.lags_model = core_dobs.lags_model
core.pics_model = core_dobs.pics_model
core.lags_sv_curve = core_dobs.lags_sv_curve
core.sv_curve = core_dobs.sv_curve

core.find_sort_d(max_dist = 11000)
core.max_cloud = len(core.sort_d)

core.C_sv = core.target_var - core.varioLUT(core.a_sv, core.C0, core.C1, sv_model = core.model)

core.C_sv[observations_direct_loc,observations_direct_loc] += C_e_const

core.condtab(normsize=10000, quantiles = 200, 
             rangn_lim = 3.5, rangn_N = 501, rangv_lim = 2.0, rangv_N = 101, model_hist = observations_direct)


N_sim = 1000

core.run_sim(N_sim, core.grid_N, core.C_sv, None, None, None,
        None, core.data, 
        observations_direct = observations_direct, observations_direct_loc = observations_direct_loc, 
        observations_direct_e = C_e_const, use_sgs = False,
        scale_m_i = True, unit_d = False, collect_all = True,
        sense_running_error = False, notebook_style = False, save_string = nb_name, sim_stochastic = False, solve_cho = True)


core.realization_to_sh_coeff(core.r_cmb, set_nmax = shc_grid)

core.m_DSS_res = core.m_DSS - s_source.data.reshape(-1,1)


del core.C_sv
del core.m_ens
del core.CQF_dist
del core.sph_d

# SAVE RESULT
print("\nSaving job")
file_pickle = open("{}.obj".format(nb_name), "wb")
pickle.dump(core, file_pickle) #, pickle_protocol=4
file_pickle.close()
print("\nJob saved and finished")