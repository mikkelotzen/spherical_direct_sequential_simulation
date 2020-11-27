from mikkel_tools.MiClass import MiClass
import mikkel_tools.utility as mt_util
import matplotlib.pyplot as plt
import pyshtools
import scipy.linalg as spl
import pickle
import numpy as np
import mikkel_tools.GMT_tools as gt
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
            Gauss_in = np.loadtxt('mikkel_tools/models_shc/Julien_Gauss_JFM_E-8_snap.dat')

        elif self.sim_type == "surface":
            Gauss_in = np.loadtxt('mikkel_tools/models_shc/Masterton_13470_total_it1_0.glm')
 
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


    def load_swarm(self, dataset, use_obs = False, target_var = None, target_var_factor = None):
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

        self.grid_theta = data_swarm["theta"]
        self.grid_phi = data_swarm["phi"]
        self.grid_radial = data_swarm["radius"]
        self.grid_obs = data_swarm["obs"]
        self.grid_N = data_swarm["N"]

        if use_obs == True:
            self.data = self.grid_obs
            # Target statistics
            if target_var_factor is not None:
                self.target_var = target_var_factor*np.var(self.data)
            elif target_var == None:
                self.target_var = np.var(self.data)
            else:
                self.target_var = target_var
            self.target_mean_true = np.mean(self.data)
            self.target_mean = 0.0

    def generate_map(self, grid_type = "glq", target_var = None, target_var_factor = None, *args):

        # Load Gauss coefficients from data files
        if np.logical_or(self.sim_type == "core", self.sim_type == "sat"):
            Gauss_in = np.loadtxt('mikkel_tools/models_shc/Julien_Gauss_JFM_E-8_snap.dat')

        elif self.sim_type == "core_ens":
            g_ens = np.genfromtxt("mikkel_tools/models_shc/gnm_midpath.dat").T*10**9
            g_ens = g_ens[:mt_util.shc_vec_len(self.N_SH),:]
            self.ensemble_B(g_ens, nmax = self.N_SH, r_at = self.r_cmb, grid_type = "glq")
            self.m_ens = self.B_ensemble[:,0,:].copy()[:,200:]

            var_ens = np.var(self.m_ens, axis=0)

            idx_close_to_var = np.argwhere(np.logical_and(var_ens>0.9995*np.mean(var_ens), var_ens<1.0005*np.mean(var_ens)))

            g = np.ravel(g_ens[:,idx_close_to_var[-1]])

            N_SH_max = self.N_SH
            self.ens_idx = int(idx_close_to_var[-1])
            #self.g_ens = g_ens

        elif self.sim_type == "lith_ens":
            g_ens = np.load("mikkel_tools/models_shc/lithosphere_g_in_rotated.npy")
            self.lith_ens_cut = 100
            g_ens = g_ens[:mt_util.shc_vec_len(self.N_SH),::self.lith_ens_cut]
            R = mt_util.lowe_shspec(self.N_SH, self.a, self.a, g_ens)
            g_ens = g_ens[:,np.mean(R,axis=0)>5]

            self.ensemble_B(g_ens, nmax = self.N_SH, r_at = self.a, grid_type = "glq")
            self.m_ens = self.B_ensemble[:,0,:].copy()

            var_ens = np.var(self.m_ens, axis=0)

            idx_close_to_var = np.argwhere(np.logical_and(var_ens>0.95*np.mean(var_ens), var_ens<1.05*np.mean(var_ens)))

            g = np.ravel(g_ens[:,idx_close_to_var[-1]])

            N_SH_max = self.N_SH
            self.ens_idx = int(idx_close_to_var[-1])

        elif self.sim_type == "surface":
            Gauss_in = np.loadtxt('mikkel_tools/models_shc/Masterton_13470_total_it1_0.glm')

        elif self.sim_type == "separation":
            Gauss_in_core = np.loadtxt('mikkel_tools/models_shc/Julien_Gauss_JFM_E-8_snap.dat')
            Gauss_in_lithos = np.loadtxt('mikkel_tools/models_shc/Masterton_13470_total_it1_0.glm')

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

        if np.logical_and.reduce((self.sim_type != "separation", self.sim_type != "core_ens", self.sim_type != "lith_ens")):
            # Compute Gauss coefficients as vector
            g = mt_util.gauss_vector(Gauss_in, self.N_SH, i_n = 2, i_m = 3)
            N_SH_max = self.N_SH
            
        # Generate field
        self.ensemble_B(g, nmax = N_SH_max, N_mf = 2, mf = True, nmf = False, r_at = self.r_grid, grid_type = grid_type)

        self.data = self.B_ensemble[:,0]
        del self.B_ensemble

        """
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
        """

        if grid_type != "swarm":
            self.r_grid_repeat = np.ones(self.N_grid,)*self.r_grid
        
        # Target statistics
        if target_var_factor is not None:
            self.target_var = target_var_factor*np.var(self.data)
        elif target_var == None:
            self.target_var = np.var(self.data)
        else:
            self.target_var = target_var
        self.target_mean_true = np.mean(self.data)
        self.target_mean = 0.0
        self.g_prior = g
        

    def condtab(self, normsize = 1001, model_hist = False, table = 'rough', quantiles = None, 
                rangn_lim = 3.5, rangn_N = 501, rangv_lim = 2.0, rangv_N = 101, rangn_geomspace = False):
        """
        Conditional distribution table
        """
        import numpy as np
        from scipy.stats import norm, laplace
        from sklearn.preprocessing import QuantileTransformer
        
        # Linearly spaced value array with start/end very close to zero/one
        start = 1e-16 #Python min
        #start = 0.001
        linspace = np.linspace(start,1-start,normsize)
        
        # Possible model target histogram cdf/ccdf
        if model_hist == True:
            ag,bg = laplace.fit(self.data)
            mod_data = np.random.laplace(ag,bg,size=100000)
            #data_sorted = np.sort(mod_data)
            data_sorted = mod_data
        elif model_hist == "laplace":
            rv = laplace()
            self.data = laplace.rvs(loc = 0, scale=1, size=self.N_grid)
            self.target_var = np.var(self.data)
            self.target_mean = 0.0
            #data_sorted = np.sort(self.data)
            data_sorted = self.data

            set_nmax = self.grid_nmax

            C_cilm = pyshtools.expand.SHExpandGLQ(self.data.reshape(self.grid_nmax+1,2*self.grid_nmax+1), self.grid_w_shtools, self.grid_zero, [1, 1, set_nmax])
            C_index = np.transpose(pyshtools.shio.SHCilmToCindex(C_cilm))

            self.g_prior = mt_util.gauss_vector_zeroth(C_index, set_nmax, i_n = 0, i_m = 1)
            self.g_cilm = C_cilm.copy()
        elif model_hist == "ensemble":
            data_sorted = np.ravel(self.m_ens)
            data_sorted = data_sorted[0.5*np.max(np.abs(data_sorted))>np.abs(data_sorted)]
            #data_sorted = np.delete(data_sorted, np.abs(data_sorted)>np.max(np.abs(data_sorted))*0.5)
        else:
            #data_sorted = np.sort(self.data)
            data_sorted = self.data
    
        if rangn_geomspace == False:
            rangn = np.linspace(-rangn_lim,rangn_lim,rangn_N)
        else:
            rangn = np.vstack((np.geomspace(-rangn_lim,-start,int(rangn_N/2)).reshape(-1,1),np.zeros((1,1)),np.geomspace(start,rangn_lim,int(rangn_N/2)).reshape(-1,1)))

        rangv = np.linspace(start,rangv_lim,rangv_N)
            
        # Normscored local conditional distributions

        # Initialize matrices
        CQF_dist = np.zeros((len(rangn),len(rangv),len(linspace)))
        CQF_mean = np.zeros((len(rangn),len(rangv)))
        CQF_var = np.zeros((len(rangn),len(rangv)))
        
        # Perform quantile transformation
        if quantiles == None:
            quantiles = int(0.1*len(data_sorted))
        
        # QuantileTransformer setup
        qt = QuantileTransformer(n_quantiles=quantiles, random_state=None, output_distribution='normal',subsample=10e8)
        qt.fit(data_sorted.reshape(-1,1))    
        #vrg = qt.transform(data_sorted.reshape(-1,1))
        
        # Generate CQF distributions, means, and variances
        print("")
        for i in range(0,len(rangn)):
            for j in range(0,len(rangv)):
                #CQF_dist[i,j,:] = np.sort(qt.inverse_transform((norm.ppf(linspace,loc=rangn[i],scale=np.sqrt(rangv[j]))).reshape(-1,1)).ravel(),axis=0)
                CQF_dist[i,j,:] = qt.inverse_transform((norm.ppf(linspace,loc=rangn[i],scale=np.sqrt(rangv[j]))).reshape(-1,1)).ravel()
                CQF_mean[i,j] = np.mean(CQF_dist[i,j,:],axis=0,dtype=np.float64)
                CQF_var[i,j] = np.var(CQF_dist[i,j,:],axis=0,ddof=1,dtype=np.float64)
                #CQF_var[i,j] = np.var(CQF_dist[i,j,:],axis=0,ddof=0,dtype=np.float64)

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
            #cloud = (self.data[i]-self.data)**2
            cloud = 0.5*(self.data[i]-self.data)**2
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
        
        #pic_zero = 0.5*np.mean(self.cloud_sorted[:self.N_grid])
        pic_zero = np.mean(self.cloud_sorted[:self.N_grid])
        lag_zero = np.mean(self.sph_d_sorted[:self.N_grid])
        pics[0] = pic_zero
        lags[0] = lag_zero
        
        lags_geom = np.linspace(self.N_grid+2, max_cloud, n_lags, dtype=int)
        for n in np.arange(0,n_lags-2):
            
            #pic = 0.5*np.mean(self.cloud_sorted[lags_geom[n]:lags_geom[n+1]:1])
            pic = np.mean(self.cloud_sorted[lags_geom[n]:lags_geom[n+1]:1])
            
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
            #sv_model = C0 + C1*(1-np.exp(-h/a))
            #sv_model = C0 + C1*(np.exp(-h/a))
            
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


    def semivar(self, model_lags = 'all', model = 'nested_sph_exp_gau', max_dist = 11000, lag_length = 5, 
                nolut = False, bounds = True, zero_nugget = False, set_model = False, hit_target_var = False):
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
                elif hit_target_var == True:
                    def semivar_return(lags_model, a):
                        return (1.5*lags_model/a-0.5*(lags_model/a)**3) 
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
                        #return C0 + C1*(1-np.exp(-lags_model/a))
                        #return C0 + C1*(np.exp(-lags_model/a))
                elif hit_target_var == True:
                    def semivar_return(lags_model, a):
                        return (1-np.exp(-3*lags_model/a))   
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
                    elif hit_target_var == True:
                        p0 = [np.max(pics_model)]
                        bounds = (0, [np.max(pics_model)])
                    else:
                        p0 = [np.mean(lags_model[-int(len(lags_model)/4.0)]),np.max(pics_model)]
                        bounds = (0, [lags_model[-1], np.max(pics_model)])
        
                if hit_target_var == True:
                    pics_model_in = pics_model/self.target_var
                    popt, pcov = curve_fit(semivar_return, lags_model, pics_model_in, bounds=bounds, p0 = p0)
                else:
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
            elif hit_target_var == True:
                C0 = 0.0 # FOR ZERO NUGGET
                C1 = self.target_var
                C2 = None
                C3 = None
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


    def cov_model(self, r_at = None, N_cut = 200):

        if r_at == None:
            r_at = self.a

        #tap_to = tap_to + 1 # One extra for overlap between R_add and R
        #n_tap = self.N_SH + tap_to - 1 # And one less in the sum as a result

        # g ensemble and parameters
        if self.sim_type == "core_ens":
            g_ens = np.genfromtxt("mikkel_tools/models_shc/gnm_midpath.dat").T*10**9
        elif self.sim_type == "lith_ens":
            g_ens = np.load("mikkel_tools/models_shc/lithosphere_g_in_rotated.npy")

        g_ens = g_ens[:mt_util.shc_vec_len(self.N_SH),:]

        if self.sim_type == "core_ens":
            g_cut = g_ens[:self.N_SH*(2+self.N_SH),N_cut:] # Truncate g
        elif self.sim_type == "lith_ens":
            g_cut = g_ens[:self.N_SH*(2+self.N_SH),::self.lith_ens_cut]

        R = mt_util.lowe_shspec(self.N_SH, r_at, self.a, g_cut)
        R = R[:,np.mean(R,axis=0)>5]

        # Angular distance matrix
        c_angdist = np.cos(mt_util.haversine(1, self.grid_phi.reshape(1,-1), 90-self.grid_theta.reshape(1,-1), 
                              self.grid_phi.reshape(-1,1), 90-self.grid_theta.reshape(-1,1)))
        c_unique, c_return = np.unique(np.ravel(c_angdist), return_inverse = True)

        # Compute constants based on Chris' note eqn. 11
        C_const = (np.arange(1,self.N_SH+1)+1)/(2*np.arange(1,self.N_SH+1)+1)
        
        # Constant and R
        CR = C_const.reshape(-1,1)*R

        # Generate matrix of all required Schmidt semi-normalized legendre polynomials
        Pn = []
        for cmu in c_unique:
            Pn.append(pyshtools.legendre.PlSchmidt(self.N_SH,cmu)[1:].reshape(-1,))
        Pn = np.array(Pn)[:,:]

        #Pn = np.array(Pn).reshape((c_angdist.shape[0],c_angdist.shape[1],-1))

        # Determine covariance model according to eqn. 11
        C_Br_model = np.mean(Pn@CR,axis=1)[c_return].reshape((c_angdist.shape[0],c_angdist.shape[1]))
        #if c_angdist.shape[0] <= 2000:
        #    C_Br = Pn@CR
        #    C_Br_model = np.mean(C_Br,axis=2)
        #else:
        #    C_Br = np.zeros((self.grid_N, self.grid_N, 1))
        #    for i in np.arange(0,R.shape[1]):
        #        C_Br += Pn@CR[:,[i]]
        #    C_Br_model = C_Br[:,:,0]/R.shape[1]

        # Positive definite covariance?
        core_eigval = spl.eigh(C_Br_model, eigvals_only=True)
        N_neg_eigval = len(core_eigval[core_eigval<=0])
        print("All eigenvalues > 0:", np.all(core_eigval>=0))
        print("Cov model is pos def:", mt_util.is_pos_def(C_Br_model))
        if np.all(core_eigval>=0) == False:
            print("Number of negative eigenvalues:",N_neg_eigval,"/",len(core_eigval))

        # Save covariance model variable
        self.C_ens = C_Br_model


    def cov_model_taper(self, r_at = None, tap_to = 500, tap_exp_p1 = 5, tap_exp_p2 = 2,
                        tap_scale_start = 0, tap_scale_end = 24, plot_taper = False, 
                        save_fig = False, save_string = "", save_dpi = 300, N_cut = 200):

        if r_at == None:
            r_at = self.a

        tap_to = tap_to + 1 # One extra for overlap between R_add and R
        n_tap = self.N_SH + tap_to - 1 # And one less in the sum as a result

        # g ensemble and parameters
        if self.sim_type == "core_ens":
            g_ens = np.genfromtxt("mikkel_tools/models_shc/gnm_midpath.dat").T*10**9
        elif self.sim_type == "lith_ens":
            g_ens = np.load("mikkel_tools/models_shc/lithosphere_g_in_rotated.npy")

        g_ens = g_ens[:mt_util.shc_vec_len(self.N_SH),:]

        if self.sim_type == "core_ens":
            g_cut = g_ens[:self.N_SH*(2+self.N_SH),N_cut:] # Truncate g
        elif self.sim_type == "lith_ens":
            g_cut = g_ens[:self.N_SH*(2+self.N_SH),::self.lith_ens_cut]

        R = mt_util.lowe_shspec(self.N_SH, r_at, self.a, g_cut)
        R = R[:,np.mean(R,axis=0)>5]

        # Angular distance matrix
        c_angdist = np.cos(mt_util.haversine(1, self.grid_phi.reshape(1,-1), 90-self.grid_theta.reshape(1,-1), 
                              self.grid_phi.reshape(-1,1), 90-self.grid_theta.reshape(-1,1)))
        c_unique, c_return = np.unique(np.ravel(c_angdist), return_inverse = True)

        # Compute covariances based on Chris' note eqn. 11
        C_const = (np.arange(1,n_tap+1)+1)/(2*np.arange(1,n_tap+1)+1)
        
        # Generate matrix of all required Schmidt semi-normalized legendre polynomials
        Pn = []
        for cmu in c_unique:
            Pn.append(pyshtools.legendre.PlSchmidt(n_tap,cmu)[1:].reshape(-1,))
        Pn = np.array(Pn)[:,:]

        #Pn = np.array(Pn).reshape((c_angdist.shape[0],c_angdist.shape[1],-1))
        
        # Define taper with inverse powered exponential sum
        lin_exp = np.linspace(tap_scale_start, tap_scale_end, tap_to)
        tap_exp = (0.5*np.exp(-tap_exp_p1*lin_exp) + 0.5*np.exp(-tap_exp_p2*lin_exp)).reshape(-1,1)

        # Take taper as factor on last spectra values and add to true prior spectra
        R_add = R[-1,:]*tap_exp
        R_tap = np.vstack((R,R_add[1:,:]))

        # Constant and R
        CR = C_const.reshape(-1,1)*R_tap

        # Determine covariance model according to eqn. 11
        C_Br_model = np.mean(Pn@CR,axis=1)[c_return].reshape((c_angdist.shape[0],c_angdist.shape[1]))
        #if c_angdist.shape[0] <= 2000:
        #    C_Br = Pn@CR
        #    C_Br_model = np.mean(C_Br,axis=2)
        #else:
        #    C_Br = np.zeros((self.grid_N, self.grid_N, 1))
        #    for i in np.arange(0,R.shape[1]):
        #        C_Br += Pn@CR[:,[i]]
        #    C_Br_model = C_Br[:,:,0]/R.shape[1]

        # Positive definite covariance?
        core_eigval = spl.eigh(C_Br_model, eigvals_only=True)
        N_neg_eigval = len(core_eigval[core_eigval<=0])
        print("All eigenvalues > 0:", np.all(core_eigval>=0))
        print("Cov model is pos def:", mt_util.is_pos_def(C_Br_model))
        if np.all(core_eigval>=0) == False:
            print("Number of negative eigenvalues:",N_neg_eigval,"/",len(core_eigval))

        # Save covariance model variable
        self.C_ens_tap = C_Br_model

        # Generate plot to show taper
        if plot_taper == True:
            lin_exp = np.linspace(tap_scale_start,tap_scale_end,10000)
            lin_deg = np.linspace(1,tap_to,10000)
            tap_exp = (0.5*np.exp(-tap_exp_p1*lin_exp) + 0.5*np.exp(-tap_exp_p2*lin_exp)).reshape(-1,1)
            R_show = R[-1,:]*tap_exp

            # Spectra
            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            for i in np.arange(R_tap.shape[1]):
                if i == 0:
                    axes[0].plot(np.arange(1,n_tap+1),R_tap[:,i],color=(0.6,0.6,0.6),label="Tapered ensemble")
                    axes[0].plot(lin_deg+self.N_SH-1,R_show[:,self.ens_idx],zorder = 10, label ="Taper function for highlight")
                    axes[0].plot(np.arange(1,n_tap+1)[:self.N_SH],R_tap[:self.N_SH,self.ens_idx],"o",zorder = 11, label = "Ensemble highlight truth")
                    axes[0].plot(np.arange(1,n_tap+1)[self.N_SH:],R_tap[self.N_SH:,self.ens_idx],"o",zorder = 11, label = "Ensemble highlight taper")
                    
                    axes[1].plot(np.arange(1,n_tap+1),R_tap[:,i],color=(0.6,0.6,0.6),label="Tapered ensemble")
                    axes[1].plot(lin_deg+self.N_SH-1,R_show[:,self.ens_idx],zorder = 10, label ="Taper function for highlight")
                    axes[1].plot(np.arange(1,n_tap+1)[:self.N_SH],R_tap[:self.N_SH,self.ens_idx],"o",zorder = 11, label = "Ensemble highlight truth")
                    axes[1].plot(np.arange(1,n_tap+1)[self.N_SH:],R_tap[self.N_SH:,self.ens_idx],"o",zorder = 11, label = "Ensemble highlight taper")
                else:
                    axes[0].plot(np.arange(1,n_tap+1),R_tap[:,i],color=(0.6,0.6,0.6))
                    axes[1].plot(np.arange(1,n_tap+1),R_tap[:,i],color=(0.6,0.6,0.6))

            
            axes[0].set_xlim(self.N_SH-5,self.N_SH+10)
            #axes[0].set_ylim(0,1.5*10**10)
            axes[0].set_ylim(0,1.2*np.max(R_tap[self.N_SH,:]))
            axes[1].set_xlim(0,tap_to/2)
            #axes[1].set_ylim(0, 10**10)
            axes[1].set_ylim(0, np.max(R_tap[self.N_SH,:]))
            axes[0].legend(fontsize="small")
            axes[1].legend(fontsize="small")
            axes[0].set_ylabel("Power [$nT^2$]")
            axes[0].set_xlabel("SH degree, n")
            axes[1].set_ylabel("Power [$nT^2$]")
            axes[1].set_xlabel("SH degree, n")
            fig.suptitle('Taper function: $f_t = 0.5e^{{-{}n}} + 0.5e^{{-{}n}}$'.format(tap_exp_p1, tap_exp_p2), fontsize=10)
            if save_fig == True:
                fig.savefig('cov_taper_{}.pdf'.format(save_string), bbox_inches='tight', dpi = save_dpi)
            plt.show()


    def sv_m_DSS(self,N,N_sim,m_DSS,sort_d,n_lags,max_cloud):

        """
        NEW Function for calculating semivariogram from simulations by taking the mean of
        equidistant lags
        """

        pics_m_DSS = np.zeros([n_lags-1,N_sim])
        for j in np.arange(0,N_sim):
            cloud_all = np.zeros([N,N])
            for i in np.arange(0,N):
                cloud = 0.5*(m_DSS[i,j]-m_DSS[:,j])**2
                cloud_all[i,:] = cloud

            pics_c = np.zeros(n_lags-1)
            cloud_ravel = np.ravel(cloud_all)[sort_d]

            pic_zero = np.mean(cloud_ravel[:N])
            #pic_zero = 0.5*np.mean(cloud_ravel[:N])
            pics_c[0] = pic_zero

            lags_geom = np.linspace(N+2,max_cloud,n_lags,dtype=int)

            for n in np.arange(0,n_lags-2):

                #pic = 0.5*np.mean(cloud_ravel[lags_geom[n]:lags_geom[n+1]:1])
                pic = np.mean(cloud_ravel[lags_geom[n]:lags_geom[n+1]:1])
                pics_c[n+1] = pic

            pics_m_DSS[:,j] = pics_c    

        self.pics_m_DSS = pics_m_DSS


    def integrating_kernel(self, obs_obj, C_e_const = 2, print_ti_est_res = False, C_mm_supply = None):

        G_mcal = mt_util.Gr_vec(self.r_grid, obs_obj.r_grid, self.lat, obs_obj.lat, self.lon, obs_obj.lon)
        self.G = np.pi/(self.grid_nmax+0.5)*np.multiply(self.grid_w,G_mcal) # +0.5 for parity with SHTOOLS

        C_e = np.diag(C_e_const**2*np.ones(obs_obj.grid_N,)) # No need to store C_e outside of here

        if C_mm_supply is None:
            self.C_mm_all = self.target_var-self.sv_lut
        else:
            self.C_mm_all = C_mm_supply

        C_dm_all = self.G*self.C_mm_all

        self.C_dd = C_dm_all*self.G.T  + C_e

        self.C_dm_all = C_dm_all.T

        self.C_e_const = C_e_const

        if print_ti_est_res == True:
            # Compute forward and get residuals to synthetic observations
            fwd_leg = self.G*self.data.reshape(-1,1)
            fwd_leg_res = obs_obj.data - fwd_leg.reshape(-1,)

            # RMSE
            rmse_leg = np.sqrt(np.mean(np.power(fwd_leg_res,2)))

            print("")
            print("Gauss-Legendre RMSE:\t %0.12f" %rmse_leg)
            plt.figure()
            y,binEdges=np.histogram(fwd_leg_res,bins=200)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            plt.plot(bincenters,y,'C0',label="Gauss-Legendre")
            plt.xlabel("Radial field residuals [nT]")
            plt.ylabel("Count")
            plt.legend()
            plt.show()


    def covmod_lsq_equiv(self, obs, C_mm, G, r_at, geomag_scale = True):      
        obs = obs.reshape(-1,1)
        C_e = np.zeros((len(obs),len(obs)))
        C_e[np.arange(1,len(obs)),np.arange(1,len(obs))] = self.C_e_const**2
        S = C_e + G@self.C_mm_all@G.T
        T = np.linalg.inv(S)
        self.m_equiv_lsq = self.C_mm_all@G.T@T@obs
        
        self.lsq_equiv_pred = G@self.m_equiv_lsq
        self.lsq_equiv_res = obs - self.lsq_equiv_pred

        self.g_equiv_lsq, _ = mt_util.sh_expand_glq(self.m_equiv_lsq, self.grid_nmax, self.grid_w_shtools, self.grid_zero, self.N_SH, geomag_scale = geomag_scale, geomag_r_at = r_at)

        #C_cilm = pyshtools.expand.SHExpandGLQ(self.m_equiv_lsq.reshape(self.grid_nmax+1,2*self.grid_nmax+1), self.grid_w_shtools, self.grid_zero, [2, 1, self.grid_nmax])

        #C_index = np.transpose(pyshtools.shio.SHCilmToCindex(C_cilm))

        #if geomag_scale == True:
        #    nm_C = mt_util.array_nm(self.grid_nmax)
        #    C_corr_sh = 1/(nm_C[:,[0]]+1)*1/(self.a/r_at)**(nm_C[:,[0]]+2)
        #    C_index = C_index[1:,:]*C_corr_sh
        #else:
        #    C_index = C_index[1:,:]

        #C_vec = mt_util.gauss_vector(C_index, self.grid_nmax, i_n = 0, i_m = 1)
        
        #self.g_lsq_equiv = C_vec


    def covmod_lsq_equiv_sep(self, obs, semivar_c, semivar_l, target_var_c, target_var_l, G_d_sep, 
                        title="", errorvar = 3**2):
        
        d_0 = obs

        G = G_d_sep.copy()
        
        C_M_c = target_var_c - semivar_c
        C_M_l = target_var_l - semivar_l

        C_M = np.zeros((G.shape[1],G.shape[1]))

        C_M[:C_M_c.shape[0],:C_M_c.shape[0]] = C_M_c
        C_M[-C_M_l.shape[0]:,-C_M_l.shape[0]:] = C_M_l
        
        C_D = np.zeros((len(d_0),len(d_0)))
        C_D[np.arange(1,len(d_0)),np.arange(1,len(d_0))] = errorvar
        S = C_D + G*C_M*G.T
        T = np.linalg.inv(S)
        m_equiv_lsq = C_M*G.T*T*d_0
        
        
        lsq_equiv_pred = G_d_sep*m_equiv_lsq
        lsq_equiv_res = obs - lsq_equiv_pred
        return m_equiv_lsq, lsq_equiv_pred, lsq_equiv_res


    def conditional_lookup(self, mu_k, sigma_sq_k, dm, dv, unit_d = False, scaling = True, return_idx = False):
        from scipy.stats import norm
        #conditional_lookup(self, cond_mean, cond_var, cond_dist, cond_dist_size, mu_k, sigma_sq_k, dm, dv):
        #conditional_lookup(core.CQF_mean, core.CQF_var, core.CQF_dist, core.condtab_normsize, mu_k, sigma_sq_k, dm_c, dv_c)

        #dist = np.power((condtab["CQF mean"]-mu_k)/dm,2)+np.power((condtab["CQF var"]-sigma_sq_k)/dv,2)
        if unit_d == True:        
            distance = np.power((self.CQF_mean-mu_k),2)+abs(self.CQF_var-sigma_sq_k)
        else:
            #distance = np.power((self.CQF_mean-mu_k)/dm,2)+abs(self.CQF_var-sigma_sq_k)/np.sqrt(dv)
            distance = abs(self.CQF_mean-mu_k)/dm+abs(self.CQF_var-sigma_sq_k)/dv

        nearest = np.unravel_index(np.argmin(distance),self.CQF_mean.shape)
        idx_n = nearest[0]
        idx_v = nearest[-1]

        m_i = self.CQF_dist[idx_n,idx_v,np.random.randint(0,self.condtab_normsize,size=1)]

        if scaling == True:
            m_i_mean = self.CQF_mean[idx_n,idx_v]        
            m_i_std = np.sqrt(self.CQF_var[idx_n,idx_v],dtype=np.float64)

            m_k = (m_i - m_i_mean)*np.sqrt(sigma_sq_k)/m_i_std+mu_k
        else:
            m_k = m_i

        if return_idx == True:
            return m_k, (idx_n, idx_v)
        else:
            return m_k


    def run_sim(self, N_sim, N_m, C_mm_all, C_dd, C_dm_all, G, observations, training_image,
                collect_all = False, scale_m_i = True, unit_d = False, sense_running_error = False, save_string = "test", solve_cho = True,
                sim_stochastic = False, separation = False, separation_lim = None, separation_obj_1 = None, separation_obj_2 = None):
                
        import time
        import random
        import scipy as sp

        """
          Input
            N_sim:
            N_m:
            prior_data:

          Output
            
        """
        
        max_cov = np.max(C_mm_all)

        """Number of simulations"""
        self.N_sim = N_sim
        m_DSS = np.zeros((N_m, N_sim))
        time_average = np.zeros((N_sim))

        """save variables"""
        self.idx_nv_collect = list()
        lagrange = list()
        self.kriging_mv_collect = list()
        rand_paths = list()
        invshapes = list()
        kriging_weights = list()
        kriging_weights_rel_dat = list()
        v_cond_vars = list()
        lstsq_param = list()
        C_dd_in = C_dd

        """ Run sequential simulations"""    
        for realization in range(0,N_sim):
            # Start timing
            t0 = time.time()
            random.seed(a=None)
            np.random.seed()

            # Initialize sequential simulation with random start
            step_rnd_path = np.arange(N_m)
            
            # Randomize index array to create random path
            random.shuffle(step_rnd_path)
            
            """Run spherical direct sequential simulation"""
            
            idx_v = np.empty([0,],dtype=int)
            idx_n = np.empty([0,],dtype=int)
            
            data_min = np.min(training_image)
            data_max = np.max(training_image)
            dm = data_max - data_min
            dv = self.target_var
            stepped_previously = np.empty([0,],dtype=int)
            
            err_mag_sum = 0.0
            len_stepped = 0
            
            # Start random walk
            for step in step_rnd_path:

                C_mm_var = C_mm_all[step,step]
                C_mm = np.empty([0,],dtype=np.longdouble)
                C_dm = np.empty([0,],dtype=np.longdouble)
                C_vm = np.empty([0,],dtype=np.longdouble)
                
                c_mm = np.empty([0,1],dtype=np.longdouble)
                c_dm = np.empty([0,1],dtype=np.longdouble)
                c_vm = np.empty([0,1],dtype=np.longdouble)
                
                mu_k = np.empty([0,],dtype=np.longdouble)
                sigma_sq_k = np.empty([0,],dtype=np.longdouble)
                idx_n = np.empty([0,],dtype=int)
                idx_v = np.empty([0,],dtype=int)
                m_i = np.empty([0,],dtype=np.longdouble)
                m_k = None
                
                err_mag_avg = np.empty([0,],dtype=np.longdouble)
                
                kriging_weights = np.empty([0,],dtype=np.longdouble)
                v_cond_var = np.empty([0,],dtype=np.longdouble)
                
                #""" SORT METHOD """

                #cov_walked = C_mm_all[step,stepped_previously]
                if np.logical_and(separation == True, step <= separation_lim):
                    sep_idx = 0
                    C_dd_in = C_dd[sep_idx]
                else:
                    sep_idx = 1
                    C_dd_in = C_dd[sep_idx]

                """COV SETUP"""

                # Set up m to m
                c_mm = C_mm_all[step,stepped_previously].reshape(-1,1)

                # Lookup all closest location semi-variances to each other (efficiently)
                C_mm = (np.ravel(C_mm_all)[(stepped_previously + (stepped_previously * C_mm_all.shape[1]).reshape((-1,1))).ravel()]).reshape(stepped_previously.size, stepped_previously.size)
            
                
                # Set up d to m
                if sim_stochastic == False:
                    c_dm = C_dm_all[step,:].reshape(-1,1)

                    if len(stepped_previously) >= 1:
                        C_dm = C_dm_all[stepped_previously,:]

                    c_vm = np.vstack((c_mm,c_dm))
                    
                    C_vm = np.zeros((len(C_dd_in)+len(C_mm),len(C_dd_in)+len(C_mm)))
                    C_vm[-len(C_dd_in):,-len(C_dd_in):] = C_dd_in
                    
                    if len(stepped_previously) >= 1:    
                        C_vm[:len(C_mm),:len(C_mm)] = C_mm
                        C_vm[:len(C_mm),-len(C_dd_in):] = C_dm
                        C_vm[-len(C_dd_in):,:len(C_mm)] = C_dm.T

                    v_cond_var = m_DSS[stepped_previously,realization].reshape(-1,1)
                    
                    if len_stepped > 0:
                        v_cond_var = np.vstack((v_cond_var,observations.reshape(-1,1)))
                    else:
                        v_cond_var = observations.reshape(-1,1)
                else:
                    if len_stepped > 1:
                        v_cond_var = m_DSS[stepped_previously,realization].reshape(-1,1)
                        c_vm = c_mm
                        C_vm = C_mm
                    else:
                        m_k = self.target_mean

                if m_k == None:
                    """SIMPLE KRIGING (SK)"""
                    #self.C_vm = C_vm
                    if solve_cho == True:
                        cho_lower = sp.linalg.cho_factor(C_vm)
                        kriging_weights = sp.linalg.cho_solve(cho_lower,c_vm)
                    else:
                        kriging_weights = np.linalg.solve(C_vm,c_vm)
                    
                    #kriging_weights[kriging_weights<0.01] = 0.0

                    #sigma_sq_k = self.target_var - np.float(kriging_weights.reshape(1,-1)@c_vm)
                    sigma_sq_k = C_mm_var - np.float(kriging_weights.reshape(1,-1)@c_vm)
                    #sigma_sq_k = max_cov - np.float(kriging_weights.reshape(1,-1)@c_vm)

                    if sigma_sq_k < 0.0:
                        print("")
                        print("Negative kriging variance: %s" %sigma_sq_k)
                        print("")
                        kriging_weights[kriging_weights<0] = 0
                        #sigma_sq_k = self.target_var - np.float(kriging_weights.reshape(1,-1)@c_vm)
                        sigma_sq_k = C_mm_var - np.float(kriging_weights.reshape(1,-1)@c_vm)
                        #sigma_sq_k = max_cov - np.float(kriging_weights.reshape(1,-1)@c_vm)
                    
                    mu_k = np.float(np.array(kriging_weights.reshape(1,-1)@(v_cond_var - self.target_mean) + self.target_mean))
                    
                    if collect_all == True:
                        if separation == True:
                            dv = C_mm_var
                            if sep_idx == 0:
                                dm = np.max(training_image[:separation_lim]) - np.min(training_image[:separation_lim])
                                m_k, idx_nv = separation_obj_1.conditional_lookup(mu_k, sigma_sq_k, dm, dv, scaling = scale_m_i, unit_d = unit_d, return_idx = True)
                            else:
                                dm = np.max(training_image[separation_lim:]) - np.min(training_image[separation_lim:])
                                m_k, idx_nv = separation_obj_2.conditional_lookup(mu_k, sigma_sq_k, dm, dv, scaling = scale_m_i, unit_d = unit_d, return_idx = True)
                        else:
                            m_k, idx_nv = self.conditional_lookup(mu_k, sigma_sq_k, dm, dv, scaling = scale_m_i, unit_d = unit_d, return_idx = True)
                        self.idx_nv_collect.append(idx_nv)
                        self.kriging_mv_collect.append((mu_k, sigma_sq_k))
                    else:
                        if separation == True:
                            dv = C_mm_var
                            if sep_idx == 0:
                                dm = np.max(training_image[:separation_lim]) - np.min(training_image[:separation_lim])
                                m_k = separation_obj_1.conditional_lookup(mu_k, sigma_sq_k, dm, dv, scaling = scale_m_i, unit_d = unit_d, return_idx = False)
                            else:
                                dm = np.max(training_image[separation_lim:]) - np.min(training_image[separation_lim:])
                                m_k = separation_obj_2.conditional_lookup(mu_k, sigma_sq_k, dm, dv, scaling = scale_m_i, unit_d = unit_d, return_idx = False)
                        else:
                            m_k = self.conditional_lookup(mu_k, sigma_sq_k, dm, dv, scaling = scale_m_i, unit_d = unit_d, return_idx = False)

                m_DSS[step,realization] = m_k
                
                # Count locations walked for search neighborhood
                stepped_previously = np.append(stepped_previously, step)
                len_stepped += 1
                
                # Get running sense of size of error compared to prior
                if sense_running_error == True:
                    err_mag = np.log10(float(np.abs((training_image)[step]-m_k)))
                    err_mag_sum += err_mag
                    err_mag_avg = float(err_mag_sum/len_stepped)
                    
                    mt_util.printProgressBar (len(stepped_previously), N_m, err_mag_avg, subject = ' realization nr. %d' % realization)
                else:
                    mt_util.printProgressBar (len(stepped_previously), N_m, subject = ' realization nr. %d' % realization)

            # End timing
            t1 = time.time()
            
            # Plot statistics of realization
            time_average[realization] = (t1-t0)
            if time_average[realization] < 60:
                print('Run time: %.3f' %(time_average[realization]), 'seconds', '')
            elif time_average[realization] < 3600:
                print('Run time: %.3f' %(time_average[realization]*60**(-1)), 'minutes', '')
            else:
                print('Run time: %.3f' %(time_average[realization]*60**(-2)), 'hours', '')
            if np.sum(time_average[:(realization+1)])*60**(-1) > 60:
                print('Total elapsed time: %.3f' %(np.sum(time_average[:(realization+1)])*60**(-2)), 'hours', '')
            else:
                print('Total elapsed time: %.3f' %(np.sum(time_average[:(realization+1)])*60**(-1)), 'minutes', '')
                
            print('Variance: %.3f' %np.var(m_DSS[:,realization]))
            print('Mean: %.3f' %np.mean(m_DSS[:,realization]))
            print('Max: %.3f' %np.max(m_DSS[:,realization]))
            print('Min: %.3f' %np.min(m_DSS[:,realization]))
            
            print('Run nr.:', realization+1)
            print('')
            
            # Save realizations after each step
            np.save("m_DSS_{}".format(save_string), m_DSS[:,:realization])

        self.m_DSS = m_DSS

        self.m_DSS_pred = G@self.m_DSS
        self.m_DSS_res = observations.reshape(-1,1) - self.m_DSS_pred

        m_DSS_mean = np.mean(self.m_DSS,axis=-1).reshape(-1,1)@np.ones((1,N_sim))
        if N_sim > 1:
            self.C_DSS = 1/(N_sim-1)*(self.m_DSS-m_DSS_mean)@(self.m_DSS-m_DSS_mean).T

        rmse_leg = np.sqrt(np.mean(np.power(self.m_DSS_res,2),axis=0))
        print("")
        print("Seqsim RMSE:\t {}".format(rmse_leg))

        color_rgb = (0.6,0.6,0.6)
        plt.figure()
        for i in np.arange(0,N_sim):
            y,binEdges=np.histogram(self.m_DSS_res[:,[i]],bins=200)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            if i == 0:
                plt.plot(bincenters,y,'-',color = color_rgb,label='Seqsim')  
            else:
                plt.plot(bincenters,y,'-',color = color_rgb)  
                
        plt.xlabel("Radial field residuals [nT]")
        plt.ylabel("Count")
        plt.show()


    def realization_to_sh_coeff(self, r_at, set_nmax = None, set_norm = 1, geomag_scale = True):
        
        #self.grid_glq(nmax = self.N_SH, r_at = r_at)

        if set_nmax == None:
            set_nmax = self.grid_nmax

        self.g_spec = []

        for i in np.arange(0,self.N_sim):
            
            C_vec, _ = mt_util.sh_expand_glq(self.m_DSS[:,[i]], self.grid_nmax, self.grid_w_shtools, self.grid_zero, set_nmax, set_norm = set_norm, geomag_scale = geomag_scale, geomag_r_at = r_at)
            
            self.g_spec.append(C_vec)

        self.g_spec = np.array(self.g_spec).T
        self.g_spec_mean = np.mean(self.g_spec,axis=1)


    def run_sim_sep(self, N_sim):
        import time
        import random

        kriging_method = "simple"

        """
        Possible kriging_method(s):
            - simple
        """

        """Number of simulations"""
        m_DSS = np.zeros((core.grid_N + lithos.grid_N, N_sim))
        time_average = np.zeros((N_sim))

        """save variables"""
        idx_nv = list()
        lagrange = list()
        kriging_mv = list()
        rand_paths = list()
        invshapes = list()
        kriging_weights = list()
        kriging_weights_rel_dat = list()
        v_cond_vars = list()
        lstsq_param = list()

        prior_data = np.hstack((core.data,lithos.data))

        """ Run sequential simulations"""    
        for realization in range(0,N_sim):
            # Start timing
            t0 = time.time()
            random.seed(a=None)
            np.random.seed()

            # Initialize sequential simulation with random start
            step_rnd_path = np.arange(core.grid_N + lithos.grid_N)
            
            # Randomize index array to create random path
            random.shuffle(step_rnd_path)
            
            """Run spherical direct sequential simulation"""
            
            idx_v = np.empty([0,],dtype=int)
            idx_n = np.empty([0,],dtype=int)
            
            data_min_c = np.min(core.data)
            data_max_c = np.max(core.data)
            dm_c = data_max_c - data_min_c
            dv_c = core.target_var
            
            data_min_l = np.min(lithos.data)
            data_max_l = np.max(lithos.data)
            dm_l = data_max_l - data_min_l
            dv_l = lithos.target_var

            stepped_previously = np.empty([0,],dtype=int)
            
            err_mag_sum_c = 0.0
            err_mag_sum_l = 0.0
            len_walked_c = 0
            len_walked_l = 0
            len_stepped = 0
            
            # Start random walk
            for step in step_rnd_path:
            
                step = step
                
                C_mm = np.empty([0,],dtype=np.longdouble)
                C_dd = np.empty([0,],dtype=np.longdouble)
                C_dm = np.empty([0,],dtype=np.longdouble)
                C_vm = np.empty([0,],dtype=np.longdouble)
                
                c_mm = np.empty([0,1],dtype=np.longdouble)
                c_dm = np.empty([0,1],dtype=np.longdouble)
                c_vm = np.empty([0,],dtype=np.longdouble)
                
                mu_k = np.empty([0,],dtype=np.longdouble)
                sigma_sq_k = np.empty([0,],dtype=np.longdouble)
                idx_n = np.empty([0,],dtype=int)
                idx_v = np.empty([0,],dtype=int)
                m_i = np.empty([0,],dtype=np.longdouble)
                m_k = np.empty([0,],dtype=np.longdouble)
                
                err_mag_avg = np.empty([0,],dtype=np.longdouble)
                
                kriging_weights = np.empty([0,],dtype=np.longdouble)
                v_cond_var = np.empty([0,],dtype=np.longdouble)
                
                """ SORT METHOD """

                cov_walked = C_mm_all[step,stepped_previously]
                
                """COV SETUP"""

                # Set up k
                c_mm = cov_walked.reshape(-1,1)
                c_dm = np.matmul(G,C_mm_all[step,:]).reshape(-1,1)
                
                # Lookup all closest location semi-variances to each other (efficiently)
                C_mm = (np.ravel(C_mm_all)[(stepped_previously + (stepped_previously * C_mm_all.shape[1]).reshape((-1,1))).ravel()]).reshape(stepped_previously.size, stepped_previously.size)
                
                # Efficient lookup of Greens
                #C_dd = GG_K_sep
                
                if len(stepped_previously) >= 1:
                    C_dm = np.matmul(G,C_mm_all[:,stepped_previously]).T
                
                c_vm = np.vstack((c_mm,c_dm))
                
                C_vm = np.zeros((len(C_dd)+len(C_mm),len(C_dd)+len(C_mm)))
                C_vm[-len(C_dd):,-len(C_dd):] = C_dd
                
                if len(stepped_previously) >= 1:    
                    C_vm[:len(C_mm),:len(C_mm)] = C_mm
                    C_vm[:len(C_mm),-len(C_dd):] = C_dm
                    C_vm[-len(C_dd):,:len(C_mm)] = C_dm.T

                v_cond_var = m_DSS[stepped_previously,realization].reshape(-1,1)
                
                if len_stepped > 0:
                    v_cond_var = np.vstack((v_cond_var,observations.reshape(-1,1))).T
                else:
                    v_cond_var = observations.reshape(-1,1).T


                if kriging_method == "simple":
                    """SIMPLE KRIGING (SK)"""

                    cho_lower = sp.linalg.cho_factor(C_vm)
                    kriging_weights = sp.linalg.cho_solve(cho_lower,c_vm)
                    
                    sigma_sq_k = C_mm_all[step,step] - np.float(kriging_weights.T*c_vm)
                    
                    if sigma_sq_k < 0.0:
                        print("")
                        print("Negative kriging variance: %s" %sigma_sq_k)
                        print("")
                        kriging_weights[kriging_weights<0] = 0
                        sigma_sq_k = C_mm_all[step,step] - np.float(kriging_weights.T*c_vm)
                    
                    mu_k = np.float(np.array(kriging_weights.T@(v_cond_var.T - 0.0) + 0.0))
                
                
                if step < core.grid_N:
                    m_k = conditional_lookup(core.CQF_mean, core.CQF_var, core.CQF_dist, core.condtab_normsize, mu_k, sigma_sq_k, dm_c, dv_c)
                else:
                    m_k = conditional_lookup(lithos.CQF_mean, lithos.CQF_var, lithos.CQF_dist, lithos.condtab_normsize, mu_k, sigma_sq_k, dm_l, dv_l)
                
                m_DSS[step,realization] = m_k
                
                # Count locations walked for search neighborhood
                stepped_previously = np.append(stepped_previously, step)
                len_stepped += 1
                
                # Get running sense of size of error compared to prior
                err_mag = np.log10(float(np.abs((prior_data)[step]-m_k)))

                if step < core.grid_N:
                    len_walked_c += 1
                    err_mag_sum_c += err_mag
                    err_mag_avg = float(err_mag_sum_c/len_walked_c)
                else:
                    len_walked_l += 1
                    err_mag_sum_l += err_mag
                    err_mag_avg = float(err_mag_sum_l/len_walked_l)
                
                mt_util.printProgressBar (len(stepped_previously), core.grid_N + lithos.grid_N, err_mag_avg, subject = ' realization nr. %d' % realization)

            # End timing
            t1 = time.time()
            
            # Plot statistics of realization
            time_average[realization] = (t1-t0)
            if time_average[realization] < 60:
                print('Run time: %.3f' %(time_average[realization]), 'seconds', '')
            elif time_average[realization] < 3600:
                print('Run time: %.3f' %(time_average[realization]*60**(-1)), 'minutes', '')
            else:
                print('Run time: %.3f' %(time_average[realization]*60**(-2)), 'hours', '')
            if np.sum(time_average[:(realization+1)])*60**(-1) > 60:
                print('Total elapsed time: %.3f' %(np.sum(time_average[:(realization+1)])*60**(-2)), 'hours', '')
            else:
                print('Total elapsed time: %.3f' %(np.sum(time_average[:(realization+1)])*60**(-1)), 'minutes', '')
                
            print('C Variance: %.3f' %np.var(m_DSS[:core.grid_N,realization]))
            print('C Mean: %.3f' %np.mean(m_DSS[:core.grid_N,realization]))
            print('C Max: %.3f' %np.max(m_DSS[:core.grid_N,realization]))
            print('C Min: %.3f' %np.min(m_DSS[:core.grid_N,realization]))

            print('L Variance: %.3f' %np.var(m_DSS[-lithos.grid_N:,realization]))
            print('L Mean: %.3f' %np.mean(m_DSS[-lithos.grid_N:,realization]))
            print('L Max: %.3f' %np.max(m_DSS[-lithos.grid_N:,realization]))
            print('L Min: %.3f' %np.min(m_DSS[-lithos.grid_N:,realization]))
            
            print('Run nr.:', realization+1)
            print('')
            
            # Save realizations after each step
            np.save("m_DSS_{}".format(nb_name), m_DSS)
    

    def pickle_save_self(self, nb_name, name_append = ""):

        del self.CQF_dist
        del self.CQF_mean
        del self.CQF_var
        del self.G
        del self.C_mm_all
        del self.C_dm_all
        del self.C_dd
        if np.logical_or(self.sim_type == "core_ens",self.sim_type == "lith_ens"):
            del self.C_ens_tap
            del self.m_ens

        # SAVE RESULT
        print("\nSaving job")
        file_pickle = open("{}{}.obj".format(nb_name, name_append), "wb")
        pickle.dump(self, file_pickle) #, pickle_protocol=4
        file_pickle.close()
        print("\nJob saved and finished")