from mikkel_tools.MiClass import MiClass
import mikkel_tools.utility as mt_util

import numpy as np
import GMT_tools as gt
import os
#import utility as sds_util

class SDSS(MiClass):   
    """ Class for performing spherical direct sequential simulation """

    def __init__(self, comment, N_SH = 60, sim_type = "core", sat_height = 350, N_SH_secondary = None):
        super().__init__(sat_height = sat_height)
        self.comment = comment
        self.class_abs_path = os.path.dirname(__file__) 

        # Initial constants related to spherical harmonics and Earth system size.
        self.N_SH = N_SH
        self.N_SH_secondary = N_SH_secondary
        self.sim_type = sim_type
        
        
    def make_grid(self, r_grid, grid, calc_sph_d = False, N_grid = 1000):
        # Initialize
        self.r_grid = r_grid
        self.grid = grid
        self.sph_d = None
        
        # Generate equal area grid
        if isinstance(grid,str):
            self.N_grid = N_grid
            N_grid_orig = self.N_grid
            check_flag = False
            if grid == "equal_area":
                while check_flag is False:
                    points_polar = mt_util.eq_point_set_polar(self.N_grid) # Compute grid with equal area grid functions
                    
                    # Set lat and lon from estimated grid
                    self.lon = points_polar[:,0]*180/np.pi
                    self.lat = 90 - points_polar[:,1]*180/np.pi
                
                    # Determine equal area grid specifics used for defining the integration area
                    s_cap, n_regions = mt_util.eq_caps(self.N_grid)
                    self.n_regions = n_regions.T
                    self.s_cap = s_cap
                    
                    if self.N_grid == int(np.sum(n_regions)):
                        check_flag = True
                        if N_grid_orig - self.N_grid != 0:
                            print("")
                            print("___ CHANGES TO GRID ___")
                            print("N = {}, not compatible for equal area grid".format(N_grid_orig))
                            print("N has been set to {}".format(self.N_grid))
                        
                    else:
                        self.N_grid -= 1
                
                self.handle_poles()
            
            # Generate Gauss-Legendre quadrature grid
            elif grid == "gauss_leg":
                self.gauss_leg_n_from_N = int(np.ceil(np.sqrt(self.N_grid/2))) # Approximate required Gauss-Legendre grid size from defined N_grid
                gauss_leg = np.polynomial.legendre.leggauss(self.gauss_leg_n_from_N) # Use built-in numpy function to generate grid
                
                # Set lat and lon range from estimated grid
                lat = 90-np.flipud(np.arccos(gauss_leg[0]).reshape(-1,1))*180/np.pi
                lon = np.arange(0,2*np.pi,np.pi/self.gauss_leg_n_from_N)*180/np.pi
                
                weights, none = np.meshgrid(gauss_leg[1],lon,indexing='ij') # Get weights for quadrature on grid
                self.weights = np.ravel(weights)
                
                # Compute full lat/lon grid
                lat, lon = np.meshgrid(lat,lon,indexing='ij')
                self.lon = lon.ravel()
                self.lat = lat.ravel()
                
                self.N_grid = 2*self.gauss_leg_n_from_N**2 # Update N_grid
                
                
            # Generate Lebedev quadrature grid
            elif grid == "lebedev":
                import quadpy
                
                # Lebedev grid generation from quadpy is limited to the following two choices
                if self.N_grid >= 5000:
                    scheme = quadpy.sphere.lebedev_131()
                else:
                    scheme = quadpy.sphere.lebedev_059()
                
                # Set lat and lon from estimated grid
                coords = scheme.azimuthal_polar
                self.lon = 180+coords[:,0]*180/np.pi
                self.lat = 90-coords[:,1]*180/np.pi
                
                self.weights = np.ravel(scheme.weights) # Get weights for quadrature on grid
                self.N_grid = len(self.weights) # Update N_grid according to Lebedev grid
        
        else:
            self.lon = grid[:,0]
            self.lat = grid[:,1]
            
            self.N_grid = len(self.lon)
            
        # Compute spherical distances between all points on grid if required
        if calc_sph_d is True:     
            lon_mesh, lat_mesh = np.meshgrid(self.lon, self.lat, indexing='ij')
            self.sph_d = mt_util.haversine(self.r_grid, lon_mesh, lat_mesh, lon_mesh.T, lat_mesh.T)


    def handle_poles(self):
        import numpy as np    
        
        # Remove the first and last grid points (the poles) and the corresponding structure related components
        idx_end_core = self.N_grid-1
        self.lat = np.delete(self.lat,[0,idx_end_core],0)
        self.lon = np.delete(self.lon,[0,idx_end_core],0)
        self.N_grid = idx_end_core-1
        
        self.n_regions = np.delete(self.n_regions,-1,1)
        self.n_regions = np.delete(self.n_regions,0,1)
        
        self.s_cap = np.delete(self.s_cap,-1,0)
        self.s_cap = np.delete(self.s_cap,0,0)
        
        self.N_grid = idx_end_core-1
        
        if self.sph_d is not None:
            self.sph_d = np.delete(self.sph_d,[0,idx_end_core],0)
            self.sph_d = np.delete(self.sph_d,[0,idx_end_core],1)


    def data(self, *args):
        
        # Generate design matrix for grid
        A_r, A_theta, A_phi = gt.design_SHA(self.r_grid/self.a, (90.0-self.lat)*self.rad, self.lon*self.rad, self.N_SH)
        G = np.vstack((A_r, A_theta, A_phi))
            
        # Load Gauss coefficients from data files
        if np.logical_or(self.sim_type == "core", self.sim_type == "sat"):
            Gauss_in = np.loadtxt('sh_models/Julien_Gauss_JFM_E-8_snap.dat')

        elif self.sim_type == "surface":
            Gauss_in = np.loadtxt('sh_models/Masterton_13470_total_it1_0.glm')
 
        else:
            Gauss_in = np.loadtxt(args[0], comments='%')
        
        # Compute Gauss coefficients as vector
        g = mt_util.gauss_vector(Gauss_in, self.N_SH, i_n = 2, i_m = 3)
        
        # Generate field data
        #data_dynamo = np.matrix(G)*np.matrix(g).T
        data_dynamo = np.matmul(G,g.T)
        data = np.array(data_dynamo[:len(A_r)]).ravel()
        self.data = np.zeros((self.N_grid,))
        self.data = data.copy() 
        self.r_grid_repeat = np.ones(self.N_grid,)*self.r_grid
        
        # Target statistics
        self.target_var = np.var(self.data)
        self.target_mean = 0.0


    def load_swarm(self, dataset, use_obs = False):
        # Load swarm samples
        data_swarm = {"SW_A":np.loadtxt("swarm_data/SW_A_AprilMayJune18_dark_quiet_NEC.txt",comments="%"), "SW_B":np.loadtxt("swarm_data/SW_B_AprilMayJune18_dark_quiet_NEC.txt",comments="%"), "SW_C":np.loadtxt("swarm_data/SW_C_AprilMayJune18_dark_quiet_NEC.txt",comments="%")}
        if dataset == "A":
            data_swarm = {"obs":data_swarm["SW_A"][:,13], "radius":data_swarm["SW_A"][:,1], "theta":(data_swarm["SW_A"][:,2]), "phi":data_swarm["SW_A"][:,3], "N":data_swarm["SW_A"][:,13].shape[0]}
        elif dataset == "B":
            data_swarm = {"obs":data_swarm["SW_B"][:,13], "radius":data_swarm["SW_B"][:,1], "theta":(data_swarm["SW_B"][:,2]), "phi":data_swarm["SW_B"][:,3], "N":data_swarm["SW_B"][:,13].shape[0]}
        elif dataset == "C":
            data_swarm = {"obs":data_swarm["SW_C"][:,13], "radius":data_swarm["SW_C"][:,1], "theta":(data_swarm["SW_C"][:,2]), "phi":data_swarm["SW_C"][:,3], "N":data_swarm["SW_C"][:,13].shape[0]}
        elif dataset == "ABC":
            data_swarm = {"obs":np.hstack((data_swarm["SW_A"][:,13],data_swarm["SW_B"][:,13],data_swarm["SW_C"][:,13])),
                                "radius":np.hstack((data_swarm["SW_A"][:,1],data_swarm["SW_B"][:,1],data_swarm["SW_C"][:,1])),
                                "theta":np.hstack(((data_swarm["SW_A"][:,2]),(data_swarm["SW_B"][:,2]),(data_swarm["SW_C"][:,2]))), 
                                "phi":np.hstack((data_swarm["SW_A"][:,3],data_swarm["SW_B"][:,3],data_swarm["SW_C"][:,3])), 
                                "N":np.hstack((data_swarm["SW_A"][:,13],data_swarm["SW_B"][:,13],data_swarm["SW_C"][:,13])).shape[0]}

        self.swarm_theta = data_swarm["theta"]
        self.swarm_phi = data_swarm["phi"]
        self.swarm_radius = data_swarm["radius"]
        self.swarm_obs = data_swarm["obs"]
        self.swarm_N = data_swarm["N"]

        if use_obs == True:
            self.data = self.swarm_obs
            # Target statistics
            self.target_var = np.var(self.data)
            self.target_mean_true = np.mean(self.data)
            self.target_mean = 0.0

    def generate_map(self, grid_type = "glq", *args):

        # Load Gauss coefficients from data files
        if np.logical_or(self.sim_type == "core", self.sim_type == "sat"):
            Gauss_in = np.loadtxt('sh_models/Julien_Gauss_JFM_E-8_snap.dat')

        elif self.sim_type == "surface":
            Gauss_in = np.loadtxt('sh_models/Masterton_13470_total_it1_0.glm')

        elif self.sim_type == "separation":
            Gauss_in_core = np.loadtxt('sh_models/Julien_Gauss_JFM_E-8_snap.dat')
            Gauss_in_lithos = np.loadtxt('sh_models/Masterton_13470_total_it1_0.glm')

            g_c = mt_util.gauss_vector(Gauss_in_core, self.N_SH, i_n = 2, i_m = 3)
            g_l = mt_util.gauss_vector(Gauss_in_lithos, self.N_SH_secondary, i_n = 2, i_m = 3)

            g_zip = (g_c,g_l)

            idx_zip_min = np.argmin((g_c.shape[0],g_l.shape[0]))
            idx_zip_max = np.argmax((g_c.shape[0],g_l.shape[0]))

            g = g_zip[idx_zip_max].copy()
            g[:g_zip[idx_zip_min].shape[0]] += g_zip[idx_zip_min]

            N_SH_max = np.max((self.N_SH, self.N_SH_secondary))

        else:
            Gauss_in = np.loadtxt(args[0], comments='%')

        if self.sim_type != "separation":
            # Compute Gauss coefficients as vector
            g = mt_util.gauss_vector(Gauss_in, self.N_SH, i_n = 2, i_m = 3)
            N_SH_max = self.N_SH
            
        # Generate field
        self.ensemble_B(g, nmax = N_SH_max, N_mf = 2, mf = True, nmf = False, r_at = self.r_grid, grid_type = grid_type)
        if grid_type == "glq":
            self.data = self.B_ensemble_glq[:,0]
            del self.B_ensemble_glq
        elif grid_type == "even":
            self.data = self.B_ensemble_even[:,0]
            del self.B_ensemble_even
        elif grid_type == "eqa":
            self.data = self.B_ensemble_eqa[:,0]
            del self.B_ensemble_eqa
        elif grid_type == "swarm":
            self.data = self.B_ensemble_swarm[:,0]

        if grid_type != "swarm":
            self.r_grid_repeat = np.ones(self.N_grid,)*self.r_grid
        
        # Target statistics
        self.target_var = np.var(self.data)
        self.target_mean_true = np.mean(self.data)
        self.target_mean = 0.0
        self.g_prior = g
        

    def condtab(self, normsize = 1000, model_hist = False, table = 'rough'):
        """
        Conditional distribution table
        """
        import numpy as np
        from scipy.stats import norm, laplace
        from sklearn.preprocessing import QuantileTransformer
        
        # Linearly spaced value array with start/end very close to zero/one
        start = 1e-16 #Python min
        linspace = np.linspace(start,1-start,normsize)
        
        # Possible model target histogram cdf/ccdf
        if model_hist == True:
            ag,bg = laplace.fit(self.data)
            mod_data = np.random.laplace(ag,bg,size=100000)
            data_sorted = np.sort(mod_data)
        else:
            data_sorted = np.sort(self.data)
    
        if table == 'fine':
            rangn = np.linspace(-3.5,3.5,1001)
            rangv = np.linspace(start,2.0,101)
        else:
            rangn = np.linspace(-3.5,3.5,501)
            rangv = np.linspace(start,2.0,101)
            
        # Normscored local conditional distributions
        
        # Initialize matrices
        CQF_dist = np.zeros((len(rangn),len(rangv),len(linspace)))
        CQF_mean = np.zeros((len(rangn),len(rangv)))
        CQF_var = np.zeros((len(rangn),len(rangv)))
        
        # Perform quantile transformation
        quantiles = int(0.1*len(data_sorted))
        
        # QuantileTransformer setup
        qt = QuantileTransformer(n_quantiles=quantiles, random_state=None, output_distribution='normal',subsample=10e8)
        qt.fit(data_sorted.reshape(-1,1))    
        #vrg = qt.transform(data_sorted.reshape(-1,1))
        
        # Generate CQF distributions, means, and variances
        print("")
        for i in range(0,len(rangn)):
            for j in range(0,len(rangv)):
                CQF_dist[i,j,:] = np.sort(qt.inverse_transform((norm.ppf(linspace,loc=rangn[i],scale=np.sqrt(rangv[j]))).reshape(-1,1)).ravel(),axis=0)
                CQF_mean[i,j] = np.mean(CQF_dist[i,j,:],axis=0,dtype=np.float64)
                CQF_var[i,j] = np.var(CQF_dist[i,j,:],axis=0,ddof=1,dtype=np.float64)
        
        self.CQF_dist = CQF_dist
        self.CQF_mean = CQF_mean
        self.CQF_var = CQF_var
        self.rangv = rangv
        self.rangn = rangn
        self.condtab_normsize = normsize
        self.condtab_model_hist = model_hist
        self.condtab_table = table

        #condtab = {"target variance":target_var, "target variance_dat":target_var_dat, "target mean":target_mean, "target mean_dat":target_mean_dat, "QF norm range":rangn, "QF var range":rangv, "CQF dist":CQF_dist, "CQF mean":CQF_mean, "CQF var":CQF_var, "target normscore":vrg, "compiler":setup["condtab_compiler"], "normsize":normsize, "start":start}

    def find_sort_d(self, max_dist = 2000):
        import numpy as np
        
        sph_d_ravel = self.sph_d.ravel()
        range_d = sph_d_ravel < max_dist
        idx_range = np.array(np.where(range_d == True)).ravel()
        val_range = sph_d_ravel[idx_range]
        idx_sort_val_range = np.argsort(val_range)
        self.sort_d = idx_range[idx_sort_val_range]
               

    def data_variogram(self, max_dist = 11000):
        """
        Function for calculating variogram from data
        """
        import numpy as np
        
        self.find_sort_d(max_dist = max_dist)
        
        cloud_all = np.zeros([self.N_grid, self.N_grid])
        
        for i in range(0,self.N_grid):
            cloud = (self.data[i]-self.data)**2
            cloud_all[i,:] = cloud
        
        self.cloud_sorted = cloud_all.ravel()[self.sort_d]
        self.sph_d_sorted = self.sph_d.ravel()[self.sort_d]

    def data_semivariogram(self, max_cloud, n_lags):
        """
        Function for calculating semivariogram from data by taking the mean of
        equidistant lags
        """
        import numpy as np
        
        pics = np.zeros(n_lags-1)
        lags = np.zeros(n_lags-1)
        
        pic_zero = 0.5*np.mean(self.cloud_sorted[:self.N_grid])
        lag_zero = np.mean(self.sph_d_sorted[:self.N_grid])
        pics[0] = pic_zero
        lags[0] = lag_zero
        
        lags_geom = np.linspace(self.N_grid+2, max_cloud, n_lags, dtype=int)
        for n in np.arange(0,n_lags-2):
            
            pic = 0.5*np.mean(self.cloud_sorted[lags_geom[n]:lags_geom[n+1]:1])
            
            pics[n+1] = pic
            
            lag_c = np.mean(self.sph_d_sorted[lags_geom[n]:lags_geom[n+1]:1])
            
            lags[n+1] = lag_c
        
        self.lags = lags
        self.pics = pics


    def semivariogram_model(self, h, a, C0, C1, C2 = None, C3 = None, sv_mode = 'spherical'):
        import numpy as np
        if sv_mode == 'spherical':
            '''
            Spherical model of the semivariogram
            '''
            
            hi = np.argsort(h)
            hir = np.argsort(hi)
            
            sv_model = np.zeros(len(h),dtype=np.longdouble)
    
            hs = h[hi]
            hla = hs[hs<a]
            sv_model[0:len(hla)] = C0 + C1*( 1.5*hla/a - 0.5*(hla/a)**3 )
            sv_model[len(hla):] = C0 + C1
            sv_model = sv_model[hir]
            
        elif sv_mode == 'dub_spherical':
            '''
            Spherical model of the semivariogram
            '''
            
            hi = np.argsort(h)
            hir = np.argsort(hi)
            
            sv_model = np.zeros(len(h),dtype=np.longdouble)
    
            hs = h[hi]
            hla = hs[hs<a]
            ha2 = h>C3
            sv_model[0:len(hla)] = C0 + C1*( 1.5*hla/a - 0.5*(hla/a)**3 ) + C2*( 1.5*hla/C3 - 0.5*(hla/C3)**3)
            sv_model[len(hla):] = C0 + C1 + C2*( 1.5*hs[len(hla):]/C3 - 0.5*(hs[len(hla):]/C3)**3)
            sv_model[ha2[hi]] = C0 + C1 + C2
            sv_model = sv_model[hir]   
            
        elif sv_mode == 'gaussian':
            '''
            Gaussian model of the semivariogram
            '''
            sv_model = C0 + C1*(1-np.exp(-(3*np.ravel(h))**2/a**2)) 
            
        elif sv_mode == 'exponential':
            '''
            Exponential model of the semivariogram
            '''
            import numpy as np
            
            sv_model = C0 + C1*(1-np.exp(-3*h/a))
            
        elif sv_mode == 'power':
            '''
            Power model of the semivariogram
            '''    
            
            hi = np.argsort(h)
            hir = np.argsort(hi)
            
            sv_model = np.zeros(len(h),dtype=np.longdouble)
    
            hs = h[hi]
            hla = hs[hs<a]
            sv_model[0:len(hla)] = C0 + C1*hla**a
            sv_model[len(hla):] = C0 + C1*np.array(hs[len(hla):])**a
            sv_model = sv_model[hir]
            
        elif sv_mode == 'hole':
            '''
            Hole model of the semivariogram
            '''  
            sv_model = C0 + C1*(1-np.cos(h/a*np.pi))
    
        elif sv_mode == 'hole_damp':
            '''
            Hole model of the semivariogram
            '''  
            sv_model = C0 + C1*(1-np.exp(-3*h/C2)*np.cos(h/a*np.pi))
    
        elif sv_mode == 'nested_hole_gau':
            '''
            Hole model of the semivariogram
            '''  
            
            hi = np.argsort(h)
            hir = np.argsort(hi)
            
            sv_model = np.zeros(len(h),dtype=np.longdouble)
    
            hs = h[hi]
            hla = hs[hs<a]
            sv_model[0:len(hla)] = C0 + C1*(1-np.cos(hla/a*np.pi)) + C2*(1-np.exp(-(3*hla)**2/a**2))
            sv_model[len(hla):] = C0 + C1*(1-np.cos(np.array(hs[len(hla):])/a*np.pi)) + C2*(1-np.exp(-(3*np.array(hs[len(hla):]))**2/a**2))
            sv_model = sv_model[hir]
            
        elif sv_mode == 'nested_sph_gau':
            '''
            Nested spherical and gaussian model of the semivariogram
            '''
            
            hi = np.argsort(h)
            hir = np.argsort(hi)
            
            sv_model = np.zeros(len(h),dtype=np.longdouble)
    
            hs = h[hi]
            hla = hs[hs<a]
            sv_model[0:len(hla)] = C0 + C1*( 1.5*hla/a - 0.5*(hla/a)**3 ) + C2*(1-np.exp(-(3*hla)**2/a**2))
            sv_model[len(hla):] = C0 + C1 + C2*(1-np.exp(-(3*np.array(hs[len(hla):]))**2/a**2))
            sv_model = sv_model[hir]
    
        elif sv_mode == 'nested_sph_exp':
            '''
            Nested spherical and exponential model of the semivariogram
            '''
            
            hi = np.argsort(h)
            hir = np.argsort(hi)
            
            sv_model = np.zeros(len(h),dtype=np.longdouble)
            
            hs = h[hi]
            hla = hs[hs<a]
            sv_model[0:len(hla)] = C0 + C1*( 1.5*hla/a - 0.5*(hla/a)**3 ) + C2*(1-np.exp(-(3*hla)/a))
            sv_model[len(hla):] = C0 + C1 + C2*(1-np.exp(-(3*np.array(hs[len(hla):]))/a))
            sv_model = sv_model[hir]
            
        elif sv_mode == 'nested_exp_gau':
            '''
            Nested exponential and gaussian model of the semivariogram
            '''
            
            hi = np.argsort(h)
            hir = np.argsort(hi)
            
            sv_model = np.zeros(len(h),dtype=np.longdouble)
            
            hs = h[hi]
            hla = hs[hs<a]
            sv_model[0:len(hla)] = C0 + C1*(1-np.exp(-(3*hla)/a)) + C2*(1-np.exp(-(3*hla)**2/a**2))
            sv_model[len(hla):] = C0 + C1*(1-np.exp(-(3*np.array(hs[len(hla):]))/a)) + C2*(1-np.exp(-(3*np.array(hs[len(hla):]))**2/a**2))
            sv_model = sv_model[hir]
            
        elif sv_mode == 'nested_sph_exp_gau':
            '''
            Nested spherical and exponential model of the semivariogram
            '''
            
            hi = np.argsort(h)
            hir = np.argsort(hi)
            
            sv_model = np.zeros(len(h),dtype=np.longdouble)
    
            hs = h[hi]
            hla = hs[hs<a]
            sv_model[0:len(hla)] = C0 + C1*( 1.5*hla/a - 0.5*(hla/a)**3 ) + C2*(1-np.exp(-(3*hla)/a)) + C3*(1-np.exp(-(3*hla)**2/a**2))
            sv_model[len(hla):] = C0 + C1 + C2*(1-np.exp(-(3*np.array(hs[len(hla):]))/a)) + C3*(1-np.exp(-(3*np.array(hs[len(hla):]))**2/a**2))
            sv_model = sv_model[hir]
            
        else:
            print('Unknown model type')
            return
        
        return sv_model


    def varioLUT(self, a, C0, C1, C2 = None, C3 = None, sv_model = 'spherical'):
        import numpy as np
        #from SDSSIM_utility import printProgressBar
        '''
        semi-variogram LUT generation
        '''
        #vario_lut = np.longdouble(np.zeros([self.N_grid, self.N_grid]))
        vario_lut = np.double(np.zeros([self.N_grid, self.N_grid]))
        
        for i in range(0,self.N_grid):
            vario_lut[:,i] = self.semivariogram_model(self.sph_d[i,:], a, C0, C1, C2=C2, C3=C3, sv_mode=sv_model)
        return vario_lut


    def semivar(self, model_lags = 'all', model = 'nested_sph_exp_gau', max_dist = 11000, lag_length = 5, nolut = False, bounds = True, zero_nugget = False, set_model = False):
        from math import inf
        import numpy as np
        from scipy.optimize import curve_fit
        #from sklearn.preprocessing import normalize
        
        self.sv_model_lags = model_lags
        self.sv_max_dist = max_dist
        self.sv_lag_length = lag_length
        self.sv_zero_nugget = zero_nugget

        self.data_variogram(max_dist=max_dist)
        
        self.max_cloud = len(self.sort_d)
        d_max = np.max(self.sph_d_sorted)
    
        self.n_lags = int(d_max/lag_length) # lags from approx typical distance between core grid points
            
        print("____semi-variogram setup___")
        print("")
        print("Number of data used: %d" %self.max_cloud)
        print("Max data distance: %.3f km" %d_max)
        print("Lag length chosen: %.1f km" %lag_length)
        print("Number of lags: %d" %self.n_lags)
        print("Number of modelling lags:",model_lags)
        print("")    
        
        self.data_semivariogram(self.max_cloud, self.n_lags)
        
        #print('Generating semi-variogram model')
        #print("")
        
        if model_lags == 'all':
            lags_model = self.lags
            pics_model = self.pics
        else:
            lags_model = self.lags[:model_lags]
            pics_model = self.pics[:model_lags]
        
        # Set model name for plotting and logicals for model selection
        self.model_names = {'spherical':'spherical', 'dub_spherical':'double spherical', 'gaussian':'gaussian', 'exponential':'exponential', 'power':'power', 'hole':'hole', 'hole_damp':'dampened hole', 'nested_hole_gau':'hole+Gaussian', 'nested_sph_gau':'spherical+Gaussian', 'nested_sph_exp':'spherical+exponential', 'nested_exp_gau':'exponential+Gaussian', 'nested_sph_exp_gau':'spherical+exponential+Gaussian'}
        self.model_select_simple = np.logical_or.reduce((model=='nested_sph_gau', model=='nested_sph_exp', model=='nested_exp_gau', model=='nested_hole_gau', model=='hole_damp'))
        self.model_select_advanced = np.logical_or.reduce((model == 'nested_sph_exp_gau', model == 'dub_spherical'))
        
        """SET MODEL OR NOT"""
        if set_model == False:
            if model == 'spherical':
                if zero_nugget == False:
                    def semivar_return(lags_model, a, C0, C1):
                        return C0 + C1*(1.5*lags_model/a-0.5*(lags_model/a)**3)
                else:
                    def semivar_return(lags_model, a, C1):
                        return C1*(1.5*lags_model/a-0.5*(lags_model/a)**3)              
            elif model == 'dub_spherical':
                if zero_nugget == False:
                    def semivar_return(lags_model, a, C0, C1, C2, C3):
                        return C0 + C1*(1.5*lags_model/a-0.5*(lags_model/a)**3) + C2*(1.5*lags_model/C3-0.5*(lags_model/C3)**3)
                else:
                    def semivar_return(lags_model, a, C1, C2, C3):
                        return C1*(1.5*lags_model/a-0.5*(lags_model/a)**3) + C2*(1.5*lags_model/C3-0.5*(lags_model/C3)**3)
            elif model == 'gaussian':
                if zero_nugget == False:
                    def semivar_return(lags_model, a, C0, C1):
                        return C0 + C1*(1-np.exp(-(3*lags_model)**2/a**2))
                else:
                    def semivar_return(lags_model, a, C1):
                        return C1*(1-np.exp(-(3*lags_model)**2/a**2))
            elif model == 'exponential':
                if zero_nugget == False:
                    def semivar_return(lags_model, a, C0, C1):
                        return C0 + C1*(1-np.exp(-3*lags_model/a))
                else:
                    def semivar_return(lags_model, a, C1):
                        return C1*(1-np.exp(-3*lags_model/a))   
            elif model == 'power':
                if zero_nugget == False:
                    def semivar_return(lags_model, a, C0, C1):
                        return C0 + C1*lags_model**a
                else:
                    def semivar_return(lags_model, a, C1):
                        return C1*lags_model**a
            elif model == 'hole':
                def semivar_return(lags_model, a, C0, C1):
                    return C0 + C1*(1-np.cos(lags_model/a*np.pi))
            elif model == 'hole_damp':
                def semivar_return(lags_model, a, C0, C1, C2):
                    return C0 + C1*(1-np.exp(-3*lags_model/C2)*np.cos(lags_model/a*np.pi))           
            elif model == 'nested_hole_gau':
                def semivar_return(lags_model, a, C0, C1, C2):
                    return C0 + C1*(1-np.cos(lags_model/a*np.pi)) + C2*(1-np.exp(-(3*lags_model)**2/a**2))
            elif model == 'nested_sph_gau':
                def semivar_return(lags_model, a, C0, C1, C2):
                    return C0 + C1*(1.5*lags_model/a-0.5*(lags_model/a)**3) + C2*(1-np.exp(-(3*lags_model)**2/a**2))
            elif model == 'nested_sph_exp':
                def semivar_return(lags_model, a, C0, C1, C2):
                    return C0 + C1*(1.5*lags_model/a-0.5*(lags_model/a)**3) + C2*(1-np.exp(-(3*lags_model)/a))
            elif model == 'nested_exp_gau':
                if zero_nugget == False:
                    def semivar_return(lags_model, a, C0, C1, C2):
                        return C0 + C1*(1-np.exp(-(3*lags_model)/a)) + C2*(1-np.exp(-(3*lags_model)**2/a**2))
                else:
                    def semivar_return(lags_model, a, C1, C2):
                        return C1*(1-np.exp(-(3*lags_model)/a)) + C2*(1-np.exp(-(3*lags_model)**2/a**2))                           
            elif model == 'nested_sph_exp_gau':
                if zero_nugget == False:
                    def semivar_return(lags_model, a, C0, C1, C2, C3):
                        return C0 + C1*(1.5*lags_model/a-0.5*(lags_model/a)**3) + C2*(1-np.exp(-(3*lags_model)/a)) + C3*(1-np.exp(-(3*lags_model)**2/a**2))
                else:
                    def semivar_return(lags_model, a, C1, C2, C3):  # FOR ZERO NUGGET
                        return C1*(1.5*lags_model/a-0.5*(lags_model/a)**3) + C2*(1-np.exp(-(3*lags_model)/a)) + C3*(1-np.exp(-(3*lags_model)**2/a**2)) # FOR ZERO NUGGET
                
            else:
                print('wrong model type chosen')
        
            if bounds == True:
                """Bounds and start values for curve fit"""
                if model == 'power':
                    if zero_nugget == False:
                        p0 = [2.0,np.min(pics_model),np.max(pics_model)]
                        bounds = (0, [2.0, inf, inf])
                    else:
                        p0 = [2.0,np.max(pics_model)]
                        bounds = (0, [2.0, inf])
                elif np.logical_or(model=='nested_sph_gau',model=='nested_sph_exp'):
                    p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model),np.max(pics_model)]
                    bounds = (0, [lags_model[-1], inf, np.max(pics_model), np.max(pics_model)])                      
                elif model=='nested_exp_gau':
                    if zero_nugget == False:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model),np.max(pics_model)]
                        bounds = (0, [lags_model[-1], inf, np.max(pics_model), np.max(pics_model)])
                    else:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.max(pics_model),np.max(pics_model)]
                        bounds = (0, [lags_model[-1], np.max(pics_model), np.max(pics_model)])
                elif model=='nested_hole_gau':
                    p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model),np.max(pics_model)]
                    bounds = (0, [lags_model[-1], inf, np.max(pics_model), np.max(pics_model)])
                elif model=='hole_damp':
                    p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model),5*np.max(lags_model)]
                    bounds = (0, [lags_model[-1], inf, np.max(pics_model), 10*np.max(lags_model)])
                elif model == 'nested_sph_exp_gau':
                    if zero_nugget == False:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model),np.max(pics_model),np.max(pics_model)]
                        bounds = (0, [lags_model[-1], inf, np.max(pics_model), np.max(pics_model),np.max(pics_model)])
                    else:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model),np.max(pics_model)]
                        bounds = (0, [lags_model[-1], np.max(pics_model), np.max(pics_model),np.max(pics_model)])
                elif model == 'dub_spherical':
                    if zero_nugget == False:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model),np.max(pics_model),np.mean(lags_model[-int(len(lags_model)/2.0)])]
                        bounds = (0, [lags_model[-1], inf, np.max(pics_model), np.max(pics_model),lags_model[-1]])
                    else:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model),np.mean(lags_model[-int(len(lags_model)/2.0)])]
                        bounds = (0, [lags_model[-1], np.max(pics_model), np.max(pics_model),lags_model[-1]])
                else: 
                    if zero_nugget == False:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.min(pics_model),np.max(pics_model)]
                        bounds = (0, [lags_model[-1], inf, np.max(pics_model)])
                    else:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.max(pics_model)]
                        bounds = (0, [lags_model[-1], np.max(pics_model)])
        
                
                popt, pcov = curve_fit(semivar_return, lags_model, pics_model, bounds=bounds, p0 = p0)
            else:    
                popt, pcov = curve_fit(semivar_return, lags_model, pics_model, method='lm')
            
            self.lags_model = lags_model
            self.pics_model = pics_model
            
            """Calculate or define nugget"""            
            if zero_nugget == False:
                C0 = popt[1]
                C1 = popt[2]
                C2 = None
                C3 = None
                
                if self.model_select_simple:
                    C2 = popt[3]
                    
                elif self.model_select_advanced:
                    C2 = popt[3]
                    C3 = popt[4]
            else:
                C0 = 0.0 # FOR ZERO NUGGET
                C1 = popt[1] # FOR ZERO NUGGET
                C2 = None
                C3 = None
                
                if self.model_select_simple:
                    C2 = popt[2]
                    
                elif self.model_select_advanced:
                    C2 = popt[2] # FOR ZERO NUGGET
                    C3 = popt[3] # FOR ZERO NUGGET
                    
            """Calculate or define correlation length"""
            a = popt[0]
            
        else:
            a = set_model["a"]
            C0 = set_model["C0"]
            C1 = set_model["C1"]
            C2 = set_model["C2"]
            C3 = set_model["C3"]
            
        """Spherical model prediction"""
        #lags_sv_curve = np.arange(0,int(np.round(lags[-1]))) # Very weird bug when using this for Gaussian model at lengths > 15K
        self.lags_sv_curve = np.linspace(0, int(np.round(self.lags[-1])), len(self.lags))
        
        if self.model_select_simple:
            self.sv_curve = self.semivariogram_model(self.lags_sv_curve, a, C0, C1, C2 = C2, sv_mode = model)
            
        elif self.model_select_advanced:
            self.sv_curve = self.semivariogram_model(self.lags_sv_curve, a, C0, C1, C2 = C2, C3 = C3, sv_mode = model)

        else:
            self.sv_curve = self.semivariogram_model(self.lags_sv_curve, a, C0, C1, sv_mode = model)
        
        print('Semi-variogram model determined, starting LUT computation')
        print("")
        if nolut == False:
            if self.model_select_simple:
                self.sv_lut = self.varioLUT(a, C0, C1, C2 = C2, sv_model = model)
            elif self.model_select_advanced:
                self.sv_lut = self.varioLUT(a, C0, C1, C2 = C2, C3 = C3, sv_model = model)
            else:
                self.sv_lut = self.varioLUT(a, C0, C1, sv_model = model)
        
        # Set model in class
        self.model = model
        self.a_sv = a
        self.C0 = C0
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        

    def sv_zs(self,N,N_sim,zs,sort_d,n_lags,max_cloud):

        """
        NEW Function for calculating semivariogram from simulations by taking the mean of
        equidistant lags
        """

        pics_zs = np.zeros([n_lags-1,N_sim])
        for j in np.arange(0,N_sim):
            cloud_all = np.zeros([N,N])
            for i in np.arange(0,N):
                cloud = (zs[i,j]-zs[:,j])**2
                cloud_all[i,:] = cloud

            pics_c = np.zeros(n_lags-1)
            cloud_ravel = np.ravel(cloud_all)[sort_d]

            pic_zero = 0.5*np.mean(cloud_ravel[:N])
            pics_c[0] = pic_zero

            lags_geom = np.linspace(N+2,max_cloud,n_lags,dtype=int)

            for n in np.arange(0,n_lags-2):

                pic = 0.5*np.mean(cloud_ravel[lags_geom[n]:lags_geom[n+1]:1])
                pics_c[n+1] = pic

            pics_zs[:,j] = pics_c    

        self.pics_zs = pics_zs