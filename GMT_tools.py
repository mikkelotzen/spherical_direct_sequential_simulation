import numpy as np

def get_Pnm_n(nmax, m, theta):
    """ 
    Calculation of associated Legendre functions P(n,m) (Schmidt normalized)
    and its derivative dP(n,m) vrt. theta.

    Input: theta[:] co-latitude (in rad)
           nmax  maximum spherical harmonic degree
    Output: Pnm    ndarray PD with Legendre functions

    P(n,m) ==> Pnm(n,m) and dP(n,m) ==> Pnm(m,n+1)
    """
    
    costh = np.cos(theta)
    sinth = np.sqrt(1-costh**2)
    
    Pnm = np.zeros((nmax+1, nmax+2, len(theta)))
    Pnm[0][0] = 1
    Pnm[1][1] = sinth
    
    rootn = np.sqrt(np.arange(0, 2*nmax**2+1))
    
#     Recursion relations after Langel "The Main Field" (1987),
# eq. (27) and Table 2 (p. 256)
    for m in np.arange(0, nmax):
#         Pnm_tmp = np.sqrt(m+m+1)*Pnm[m][m]
         Pnm_tmp = rootn[m+m+1]*Pnm[m][m]
         Pnm[m+1][m] = costh*Pnm_tmp
         if m > 0: 
#             Pnm[m+1][m+1] = sinth*Pnm_tmp/np.sqrt(m+m+2.)
             Pnm[m+1][m+1] = sinth*Pnm_tmp/rootn[m+m+2]
         for n in np.arange(m+2, nmax+1):
             d = n*n - m*m
             e = n + n - 1
#             Pnm[n][m] = (e*costh*Pnm[n-1][m]-np.sqrt(d-e)*Pnm[n-2][m])/np.sqrt(d)
             Pnm[n][m] = (e*costh*Pnm[n-1][m]-rootn[d-e]*Pnm[n-2][m])/rootn[d]

#    dP(n,m) = Pnm(m,n+1) is the derivative of P(n,m) vrt. theta
    Pnm[0][2] = -Pnm[1][1]
    Pnm[1][2] =  Pnm[1][0]
    for n in np.arange(2, nmax+1):
          l = n + 1
          Pnm[0][l] = -np.sqrt(.5*(n*n+n))*Pnm[n][1]
          Pnm[1][l] = .5*(np.sqrt(2.*(n*n+n))*Pnm[n][0]-np.sqrt((n*n+n-2.))*Pnm[n][2])

          for m in np.arange(2, n):
              Pnm[m][l] = .5*(np.sqrt((n+m)*(n-m+1.))*Pnm[n][m-1] -np.sqrt((n+m+1.)*(n-m))*Pnm[n][m+1])

          Pnm[n][l] = .5*np.sqrt(2.*n)*Pnm[n][n-1] 

    return Pnm

def get_Pnm(nmax, theta):
    """ 
    Calculation of associated Legendre functions P(n,m) (Schmidt normalized)
    and its derivative dP(n,m) vrt. theta.

    Input: theta[:] co-latitude (in rad)
           nmax  maximum spherical harmonic degree
    Output: Pnm    ndarray PD with Legendre functions

    P(n,m) ==> Pnm(n,m) and dP(n,m) ==> Pnm(m,n+1)
    """
    
    costh = np.cos(theta)
    sinth = np.sqrt(1-costh**2)
    
    Pnm = np.zeros((nmax+1, nmax+2, len(theta)))
    Pnm[0][0] = 1
    Pnm[1][1] = sinth
    
    rootn = np.sqrt(np.arange(0, 2*nmax**2+1))
    
#     Recursion relations after Langel "The Main Field" (1987),
# eq. (27) and Table 2 (p. 256)
    for m in np.arange(0, nmax):
#         Pnm_tmp = np.sqrt(m+m+1)*Pnm[m][m]
         Pnm_tmp = rootn[m+m+1]*Pnm[m][m]
         Pnm[m+1][m] = costh*Pnm_tmp
         if m > 0: 
#             Pnm[m+1][m+1] = sinth*Pnm_tmp/np.sqrt(m+m+2.)
             Pnm[m+1][m+1] = sinth*Pnm_tmp/rootn[m+m+2]
         for n in np.arange(m+2, nmax+1):
             d = n*n - m*m
             e = n + n - 1
#             Pnm[n][m] = (e*costh*Pnm[n-1][m]-np.sqrt(d-e)*Pnm[n-2][m])/np.sqrt(d)
             Pnm[n][m] = (e*costh*Pnm[n-1][m]-rootn[d-e]*Pnm[n-2][m])/rootn[d]

#    dP(n,m) = Pnm(m,n+1) is the derivative of P(n,m) vrt. theta
    Pnm[0][2] = -Pnm[1][1]
    Pnm[1][2] =  Pnm[1][0]
    for n in np.arange(2, nmax+1):
          l = n + 1
          Pnm[0][l] = -np.sqrt(.5*(n*n+n))*Pnm[n][1]
          Pnm[1][l] = .5*(np.sqrt(2.*(n*n+n))*Pnm[n][0]-np.sqrt((n*n+n-2.))*Pnm[n][2])

          for m in np.arange(2, n):
              Pnm[m][l] = .5*(np.sqrt((n+m)*(n-m+1.))*Pnm[n][m-1] -np.sqrt((n+m+1.)*(n-m))*Pnm[n][m+1])

          Pnm[n][l] = .5*np.sqrt(2.*n)*Pnm[n][n-1] 

    return Pnm

def design_SHA(r, theta, phi, nmax):
    """
    Created on Fri Feb  2 09:18:42 2018

    @author: nilos
    A_r, A_theta, A_phi = design_SHA(r, theta, phi, N)

     Calculates design matrices A_i that connects the vector 
     of (Schmidt-normalized) spherical harmonic expansion coefficients, 
     x = (g_1^0; g_1^1; h_1^1; g_2^0; g_2^1; h_2^1; ... g_N^N; h_N^N) 
     and the magnetic component B_i, where "i" is "r", "theta" or "phi":
         B_i = A_i*x
         Input: r[:]      radius vector (in units of the reference radius a)
         theta[:]  colatitude    (in radians)
         phi[:]    longitude     (in radians)
         N         maximum degree/order

     A_r, A_theta, A_phi = design_SHA(r, theta, phi, N, i_e_flag)
     with i_e_flag = 'int' for internal sources (g_n^m and h_n^m)
                     'ext' for external sources (q_n^m and s_n^m) 
     """
    
    cml = np.zeros((nmax+1, len(theta))) # cos(m*phi)
    sml = np.zeros((nmax+1, len(theta))) # sin(m*phi)
    a_r = np.zeros((nmax+1, len(theta)))
    cml[0]= 1
    for m in np.arange(1, nmax+1):
        cml[m]=np.cos(m*phi)
        sml[m]=np.sin(m*phi)
    for n in np.arange(1, nmax+1):
        a_r[n]=r**(-(n+2)) 

    Pnm = get_Pnm(nmax, theta)
    sinth = Pnm[1][1]

# construct A_r, A_theta, A_phi
    A_r =     np.zeros((nmax*(nmax+2), len(theta)))
    A_theta = np.zeros((nmax*(nmax+2), len(theta)))
    A_phi =   np.zeros((nmax*(nmax+2), len(theta)))
    
    l = 0
    for n in np.arange(1, nmax+1):
        for m in np.arange(0, n+1):
            A_r[l] =     (n+1.)*Pnm[n][m]   *cml[m] * a_r[n]
            A_theta[l] =       -Pnm[m][n+1] *cml[m] * a_r[n]
            A_phi[l] =        m*Pnm[n][m]   *sml[m] * a_r[n] / sinth
            l=l+1
            if m > 0: 
                A_r[l] =     (n+1.)*Pnm[n][m]   * sml[m] * a_r[n]
                A_theta[l] =       -Pnm[m][n+1] * sml[m] * a_r[n]
                A_phi[l] =       -m*Pnm[n][m]   * cml[m] * a_r[n] / sinth
                l=l+1
    return A_r.transpose(), A_theta.transpose(), A_phi.transpose()

def synth_grid(gh, r, theta, phi):
    """
    Created on Fri Feb  2 09:18:42 2018

    @author: nilos
    B_r, B_theta, B_phi = synth_grid(gh, r, theta, phi)

    """
    n_coeff = len(gh)
    nmax = int(np.sqrt(n_coeff+1)-1)
    N_theta   = len(theta)
    N_phi     = len(phi)
    n = np.arange(0, nmax+1)
     
    r_n       = r**(-(n+2))
     
    cos_sin_m = np.ones((n_coeff, N_phi))
    sin_cos_m = np.zeros((n_coeff, N_phi))
    T_r     = np.zeros((n_coeff, N_theta))
    T_theta = np.zeros((n_coeff, N_theta))
    T_phi   = np.zeros((n_coeff, N_theta))

    Pnm = get_Pnm(nmax, theta)
    sinth = Pnm[1][1]

    k=0
    for n in np.arange(1, nmax+1):
        T_r[k]     = (n+1.)*r_n[n]*Pnm[n][0]
        T_theta[k] = -r_n[n]*Pnm[0][n+1]
        k = k+1
        for m in np.arange(1, n+1):
            T_r[k]     = (n+1)*r_n[n]*Pnm[n][m]
            T_theta[k] = -r_n[n]*Pnm[m][n+1]
            T_phi[k]   = m*r_n[n]*Pnm[n][m] / sinth
            cos_sin_m[k] = np.cos(m*phi)
            sin_cos_m[k+1] = cos_sin_m[k]
            T_r[k+1]     = T_r[k]
            T_theta[k+1] = T_theta[k]
            T_phi[k+1]   = -T_phi[k]
            cos_sin_m[k+1] = np.sin(m*phi)
            sin_cos_m[k] = cos_sin_m[k+1]
            k = k+2

    tmp = cos_sin_m*gh[:, np.newaxis]
    B_r = np.matmul(T_r.transpose(), tmp)
    B_theta = np.matmul(T_theta.transpose(), tmp)
    B_phi = np.matmul(T_phi.transpose(), sin_cos_m*gh[:, np.newaxis])

    return B_r, B_theta, B_phi

def read_shc(shc_fn,cols='all'):
  """
  Read values of Gauss coefficients (g,h) from column(s) in file.

  File should be ascii file obeying the SHC format.

  Parameters
  ----------
  shc_fn : str
    Path of input SHC ascii file
  cols : list_like
    List of columns to read from file. This should correspond to the
    columns the different times values coefficients will be 
    read from. In a standard SHC file the first two columns (0 and 1)
    correspond to the degree (l) and order (m) of the harmonic and
    should not be included in `cols`. As such the default value 
    ``cols='all'`` corresponds to ``cols=range(2,2+N_times)``, where 
    ``N_times`` is the number of time snapshots in the file.  

  Returns
  -------
  Tuple
    Tuple with following values at given indices:

      0. numpy.ndarray of gaussian coefficients with such that 
          ``myarray[0]`` gives all coefficients at the first time point,
          given that there are multiple time snapshots. Otherwise 
          ``array[0]`` will only contain the first coefficient.
      1. spline order `k` as an integer used to reconstruct model from 
          time snapshots.
      2. number of columns as an integer.
      3. time of the temporal snapshots (in fractional years in the 
          standard SHC format) as 1D `numpy.ndarray`.

  Notes
  -----
  Missing data values marked as NaN are currently not handled.
  
  """
 
  with open(shc_fn) as f:
    headerlen=0
    header_fin=False
    for line in f:
      if header_fin: # finished reading header
        times=np.empty(N_times)
        c=0
        for t in line.split():
          times[c]=float(t)
          c+=1
        break
      else:
        if line.startswith('#'):
          headerlen+=1
        else:
          header_fin=True
          N_min,N_max,N_times,spline_order,N_step = \
            (int(v) for v in line.split()[:5])
  cols=range(2,2+N_times)
#  gh=np.loadtxt(shc_fn,skiprows=headerlen+2)
  gh=np.loadtxt(shc_fn,skiprows=headerlen+2,usecols=cols,unpack=True)
#  if len(gh.shape)==1:
#    gh=np.expand_dims(gh,0)
  
  # currently not passing on N_min,N_max,N_step
  return gh,spline_order,times

def mauersberger_lowes_spec(gh,r=1):
    """ The Mauersberger-Lowes spatial powerspectrum"""
    ratio=1/r
    N = int(np.sqrt(gh.size+1)-1) # maximum spherical harmonic degree
    R_l=np.empty(N)
    gh_idx=0
    for l in range(1,N+1):
        gh_idx_n=gh_idx+2*l+1
        g_sq=np.sum(gh[gh_idx:gh_idx_n]**2)
        R_l[l-1] = (l+1)*ratio**(2*l+4)*g_sq
        gh_idx=gh_idx_n        
    return R_l


def degree_correlation(gh1,gh2,lmax=-1,lmin=1):
    """Correlation per spherical harmonic degree between two models 1 and 2"""
    if lmax<1:
        lmax1,lmin1=get_l_maxmin(len(gh1))
        lmax2,lmin2=get_l_maxmin(len(gh2))
        lmax = min(lmax1,lmax2)
        lmin = min(lmin1,lmin2)
    c12=np.empty(lmax+1-lmin)
    i=0
    for l in range(lmin,lmax+1):
        #m=0
        g12 = gh1[i]*gh2[i]
        g11 = gh1[i]**2
        g22 = gh2[i]**2
        i+=1
        for m in range(1,l+1):
            g12 += gh1[i]*gh2[i]
            g11 += gh1[i]**2
            g22 += gh2[i]**2
            i += 2
        c12[l-lmin] = g12/np.sqrt(g11*g22)
    return c12


