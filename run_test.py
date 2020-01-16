#!/usr/bin/env python3

from sdss_stochastic import SDSS_stochastic

#%%

core = SDSS_stochastic(sim_type = "core")

#%%
core.grid(core.r_cmb, "equal_area", calc_sph_d = True)
#core.grid(core.r_cmb, "gauss_leg", calc_sph_d = True)
#core.grid(core.r_cmb, "lebedev", calc_sph_d = True)

#%%

core.data()

#%%

core.condtab()

#%%

core.semivar(model_lags = 'all', model = 'dub_spherical', lag_length = 5)

#%%

