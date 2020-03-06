"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Utility functions for use in lithosphere_prior

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt

def dict_save(path, name, variable ):
    import pickle
    with open('%s' %path + name + '.pkl', 'wb') as f:
        pickle.dump(variable, f, pickle.HIGHEST_PROTOCOL)

def dict_load(path, name ):
    import pickle
    with open('%s' %path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def variable_save(filename, variable):
    import numpy as np
    
    np.save(filename,variable) # Save variable
    print('Saved file:', filename) 

def variable_load(filename):
    import numpy as np
    
    variable = np.load(filename) # Load variable
    
    print('Finished loading:', filename)
    return np.array(variable) 

# Function for computing a vector of Gauss coefficicents given standard input
def gauss_vector(g_in, N_deg, i_n = 0, i_m = 1):
    import numpy as np
    
    i=0
    i_line=0
        
    g = np.zeros(2*np.sum(np.arange(1,N_deg+1)+1) - N_deg)
    
    for n in range(1,N_deg+1):
        for m in range(0,n+1):
            if m == 0: 
                g[i]=g_in[i_line,i_n]
                i += 1
                i_line += 1            
            else:
                g[i]=g_in[i_line,i_n]
                g[i+1]=g_in[i_line,i_m]
                i+= 2  
                i_line += 1
                
    return g

def printProgressBar (iteration, total, *args, subject='', prefix = '', suffix = '', decimals = 1, length = 10, fill = 'O'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    
    if args:
        print('\r%s |%s| %s%% %s %s. Counter: %s/%s, Running error magnitude: %.1f' % (prefix, bar, percent, suffix, subject, iteration, total, args[0]), end = '\r')
    else:
        print('\r%s |%s| %s%% %s %s. Counter: %s/%s' % (prefix, bar, percent, suffix, subject, iteration, total), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def plot_cartopy_global(lat, lon, data=None, unit = "[nT]", cmap = 'PuOr_r', vmin=None, vmax=None, figsize=(8,8), title='Cartopy Earth plot', lat_0 = 0.0, lon_0 = 0.0, point_size=10, showfig=True, norm_class = False, scale_uneven = False, fill = False, savefig = False, dpi = 100, path = None, saveformat = ".png"):

    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    #from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    import numpy as np
    import matplotlib.colors as colors

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    class SqueezedNorm(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, mid=0, s1=1.75, s2=1.75, clip=False):
            self.vmin = vmin # minimum value
            self.mid  = mid  # middle value
            self.vmax = vmax # maximum value
            self.s1=s1; self.s2=s2
            f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
            self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                                 f(x,zero,vmin,s2)*(x<zero)+0.5
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
            return np.ma.masked_array(r)

    #fig = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    
    projection = ccrs.Mollweide()
    axes_class = (GeoAxes, dict(map_projection=projection))
    
    fig = plt.figure(figsize=figsize)
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(1, 1),
                    axes_pad=0.1,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0.05,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode
    #if fill is True:

    if data is None:
        axgr[0].scatter(lon, lat, s=point_size, transform=ccrs.PlateCarree(), cmap=cmap)

    else:
        if vmin is None:
            vmin = np.min(data)
            vmax = np.max(data)
            
        if scale_uneven == False:
            veven = np.max([abs(vmax),abs(vmin)])

            cb = axgr[0].scatter(lon, lat, s=point_size, c=data, transform=ccrs.PlateCarree(), vmin = -veven, vmax = veven, cmap=cmap)	
        else:
            scale_diff = vmax-vmin
            if norm_class == "midpoint":
                cb = axgr[0].scatter(lon, lat, s=point_size, c=data, transform=ccrs.PlateCarree(), vmin = (vmax - scale_diff), vmax = vmax, cmap=cmap, norm=MidpointNormalize(midpoint=0.))
            elif norm_class == "squeezed":
                cb = axgr[0].scatter(lon, lat, s=point_size, c=data, transform=ccrs.PlateCarree(), vmin = (vmax - scale_diff), vmax = vmax, cmap=cmap, norm=SqueezedNorm())
            else:
                cb = axgr[0].scatter(lon, lat, s=point_size, c=data, transform=ccrs.PlateCarree(), vmin = (vmax - scale_diff), vmax = vmax, cmap=cmap)
        #plt.colorbar(cb,location='bottom',pad="5%",size="5%").set_label(label='%s %s' %(title,unit), size=20, weight='bold')
        axgr.cbar_axes[0].colorbar(cb)
        #axgr.cbar_axes[0].set_label('%s %s' %(title,unit))
        cax = axgr.cbar_axes[0]
        axis = cax.axis[cax.orientation]
        axis.label.set_text('%s %s' %(title,unit))

    axgr[0].coastlines()
    axgr[0].set_global()
    

    if savefig is True and path is not None:
        title_to_filename = title.replace(" ", "_").replace(":","").replace("-","_").replace("/","").replace("(","").replace(")","")
        plt.savefig('%s%s%s' %(path,title_to_filename,saveformat), bbox_inches='tight', dpi = dpi, format="%s" %saveformat.replace(".",""))
    if showfig is True:
        plt.show()
    return

def plot_cartopy_global_new(lat = None, lon = None, data=None, limits_data = None, plot_quality = None, unit = "[nT]", cmap = plt.cm.RdBu_r, projection_transformation = "Mollweide", figsize=(10,10), title='Cartopy Earth plot', lat_0 = 0.0, lon_0 = 0.0, point_size=2, showfig=True, norm_class = False, scale_uneven = False, shift_grid = False, savefig = False, dpi = 100, path = None, saveformat = ".png"):

    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    import numpy as np
    import matplotlib.colors as colors
    import matplotlib.colorbar
    
    # Start figure
    fig = plt.figure(figsize=figsize)
    
    vmin = np.min(limits_data)
    vmax = np.max(limits_data)
    
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    class SqueezedNorm(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, mid=0, s1=1.75, s2=1.75, clip=False):
            self.vmin = vmin # minimum value
            self.mid  = mid  # middle value
            self.vmax = vmax # maximum value
            self.s1=s1; self.s2=s2
            f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
            self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                                 f(x,zero,vmin,s2)*(x<zero)+0.5
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
            return np.ma.masked_array(r)
    
    # Plotting ranges and norms
    if limits_data is None:
        vmin = np.min(data)
        vmax = np.max(data)

    if scale_uneven == False:
        veven = np.max([abs(vmax),abs(vmin)])
        vmin = -veven
        vmax = veven
        norm_in = None
    else:
        scale_diff = vmax-vmin
        vmin = (vmax - scale_diff)

        if norm_class == "midpoint":
            norm_in = MidpointNormalize(midpoint=0.)
        elif norm_class == "squeezed":
            norm_in = SqueezedNorm()
        else:
            norm_in = None
    
    # Plotting init
    if projection_transformation == "ortho":
        projection = ccrs.Orthographic(central_longitude=lon_0, central_latitude=lat_0)
    else:
        projection = ccrs.Mollweide()
    
    
    if plot_quality == "high":
        axes_class = (GeoAxes, dict(map_projection=projection))
        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                        nrows_ncols=(1, 1),
                        axes_pad=0.1,
                        cbar_location='bottom',
                        cbar_mode='single',
                        cbar_pad=0.05,
                        cbar_size='5%',
                        label_mode='')  # note the empty label_mode

        axgr[0].coastlines()
        axgr[0].set_global()
        
        if data is None:
            axgr[0].scatter(lon, lat, s=point_size, transform=ccrs.PlateCarree(), cmap=cmap)

        else:
            cb = axgr[0].scatter(lon, lat, s=point_size, c=data, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax, cmap=cmap, norm = norm_in)

            axgr.cbar_axes[0].colorbar(cb)
            cax = axgr.cbar_axes[0]
            axis = cax.axis[cax.orientation]
            axis.label.set_text('%s %s' %(title,unit))
    else:
        ax = plt.axes(projection=projection)

        ax.coastlines()
        ax.set_global()

        data_in = np.flipud(np.ravel(data).reshape(360,720))
        if shift_grid == True:
            data_in = np.hstack((data_in[:,360:],data_in[:,:360]))

        cs = ax.imshow(data_in,  vmin = vmin, vmax = vmax, cmap = cmap, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])

        cax,kw = matplotlib.colorbar.make_axes(ax,location='bottom', pad=0.02, shrink=0.7, aspect=60)
        out=fig.colorbar(cs,cax=cax,extend='neither',**kw)
        out.set_label('%s %s' %(title,unit), size=10)
        ax.background_patch.set_fill(False)
        
    if savefig is True and path is not None:
        title_to_filename = title.replace(" ", "_").replace(":","").replace("-","_").replace("/","").replace("(","").replace(")","")
        plt.savefig('%s%s%s' %(path,title_to_filename,saveformat), bbox_inches='tight', dpi = dpi, format="%s" %saveformat.replace(".",""))
    if showfig is True:
        plt.show()
    return

def plot_cartopy_animation(lat = None, lon = None, data=None, limits_data = None, animation_quality = None, frames = 2, interval = 200, projection_transformation = "Mollweide", unit = "[nT]", title = "Cartopy Earth Plot", cmap = plt.cm.RdBu_r, figsize=(10,10), point_size=1, norm_class = False, scale_uneven = False, shift_grid = False, animation_output = "javascript", filename = "", path_save_mp4 = "images/"):

    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    import numpy as np
    import matplotlib.colors as colors
    import matplotlib.colorbar
    
    # animation
    from matplotlib import animation, rc
    from IPython.display import HTML, Image, display, Video
    import os
    
    if data is None:
        raise ValueError("No data accessible for animation")
    
    if animation_output == "html5":
        html = "html5"
    else:
        html = "jshtml"
        
    rc('animation', html=html)
        
    
    # Start figure
    fig = plt.figure(figsize=figsize)
    
    vmin = np.min(limits_data)
    vmax = np.max(limits_data)
    
    # COLORBAR TRANSFORM CLASSES
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
            return np.ma.masked_array(np.interp(value, x, y))

    class SqueezedNorm(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, mid=0, s1=1.75, s2=1.75, clip=False):
            self.vmin = vmin # minimum value
            self.mid  = mid  # middle value
            self.vmax = vmax # maximum value
            self.s1=s1; self.s2=s2
            f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
            self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                                 f(x,zero,vmin,s2)*(x<zero)+0.5
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
            return np.ma.masked_array(r)
    
    # Plotting ranges and norms
    if limits_data is None:
        vmin = np.min(data)
        vmax = np.max(data)

    if scale_uneven == False:
        veven = np.max([abs(vmax),abs(vmin)])
        vmin = -veven
        vmax = veven
        norm_in = None

    else:
        scale_diff = vmax-vmin
        vmin = (vmax - scale_diff)

        if norm_class == "midpoint":
            norm_in = MidpointNormalize(midpoint=0.)
        elif norm_class == "squeezed":
            norm_in = SqueezedNorm()
        else:
            norm_in = None
    
    # Plotting init
    if projection_transformation == "ortho":
        projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=0.0)
    else:
        projection = ccrs.Mollweide()
    
    
    if animation_quality == "high":
        axes_class = (GeoAxes, dict(map_projection=projection))
        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                        nrows_ncols=(1, 1),
                        axes_pad=0.1,
                        cbar_location='bottom',
                        cbar_mode='single',
                        cbar_pad=0.05,
                        cbar_size='5%',
                        label_mode='')  # note the empty label_mode

        axgr[0].coastlines()
        axgr[0].set_global()
                
        cb = axgr[0].scatter(lon, lat, s=point_size, c=limits_data, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax, cmap=cmap, norm = norm_in)

        axgr.cbar_axes[0].colorbar(cb)
        cax = axgr.cbar_axes[0]
        axis = cax.axis[cax.orientation]
        axis.label.set_text('%s %s' %(title,unit))

        def animate(i):
            cb = axgr[0].scatter(lon, lat, s=point_size, c=data[:,i], transform=ccrs.PlateCarree(), cmap=cmap, norm = norm_in)
            return (cb,)
        
    else:
        ax = plt.axes(projection=projection)

        ax.coastlines()
        ax.set_global()

        data_init = np.flipud(np.ravel(limits_data).reshape(360,720))
        if shift_grid == True:
            data_init = np.hstack((data_init[:,360:],data_init[:,:360]))

        cs = ax.imshow(data_init,  vmin = vmin, vmax = vmax, cmap = cmap, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])

        cax,kw = matplotlib.colorbar.make_axes(ax,location='bottom', pad=0.02, shrink=0.7, aspect=60)
        out=fig.colorbar(cs,cax=cax,extend='neither',**kw)
        out.set_label('%s %s' %(title,unit), size=10)
        
        def animate(i):
            data_i = data[:,i]
            data_i = np.flipud(np.ravel(data_i).reshape(360,720))
            if shift_grid == True:
                data_i = np.hstack((data_i[:,360:],data_i[:,:360]))
            cs = ax.imshow(data_i,  vmin = vmin, vmax = vmax, cmap = cmap, norm=norm_in, transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])
            return (cs,)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval)
    
    plt.close() # Close the active figure to avoid extra plot
    
  
    if animation_output == "html5":
        if os.path.exists('rm ./{}{}.mp4'.format(path_save_mp4,filename)):
            os.remove('rm ./{}{}.mp4'.format(path_save_mp4,filename))
            
        fps = int(frames/(frames*(interval/1000)))
        anim.save('{}{}.mp4'.format(path_save_mp4,filename), fps = fps, writer='ffmpeg')
        return HTML('<left><video controls autoplay loop src="{}{}.mp4?{}" width=100%/></left>'.format(path_save_mp4,filename,int(np.random.uniform(1,10e20))))
    
    else:
        return anim
        #return HTML(anim.to_jshtml())
    #return HTML(anim.to_html5_video())
    #return anim

def plot_power_spectrum(p_spec, figsize=(14,8)):
    import matplotlib.pyplot as plt
    import numpy as np
    ns = np.arange(1,len(p_spec))
    n_ticks = np.append(np.append(np.array([1,]),np.arange(10,np.max(ns),step=10)),np.max(ns))
    plt.figure(figsize=figsize)
    plt.plot(ns, p_spec[1:])
    plt.yscale('log')
    plt.xlabel("Spherical harmonic degree")
    plt.ylabel("Power [nt²]")
    plt.xticks(n_ticks, fontsize="small")
    plt.grid(alpha=0.3)
    plt.show()

def plot_ensemble_histogram(ensemble, N_ensemble, target = None, figsize=(10,10), unit = "", savefig = False, savepath = "./", filename = "file", fontsize = 10, dpi = 100):
    import numpy as np
    plt.figure(figsize=figsize)
    if N_ensemble > 1:
        for j in range(0,N_ensemble-1):
            y,binEdges=np.histogram(ensemble[:,j],bins=200)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            plt.plot(bincenters,y,'-',color = '0.75')
    else:
        j = -1
        
    y,binEdges=np.histogram(ensemble[:,j+1],bins=200)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,'-',color = '0.75',label='Ensemble')    
    
    if target is not None:
        y,binEdges=np.histogram(target,bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'k-',label='Target')

    plt.legend(loc='best',fontsize=fontsize)
    plt.xlabel('Ensemble value {}'.format(unit), fontsize=fontsize, labelpad=8)
    plt.ylabel('Bin count', fontsize=fontsize, labelpad=8)
    if savefig == True: 
        plt.savefig('{}{}.png'.format(savepath, filename), bbox_inches='tight', dpi = dpi)
    plt.show()

def haversine(radius, lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    import numpy as np
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = radius * c
    return km

def handle_poles(grid_core, setup_core, grid_sat, setup_sat):
    import numpy as np    
    
    if grid_core is not None:
        idx_end_core = grid_core["N"]-1
        grid_core["lat"] = np.delete(grid_core["lat"],[0,idx_end_core],0)
        grid_core["lon"] = np.delete(grid_core["lon"],[0,idx_end_core],0)
        grid_core["N"] = idx_end_core-1
        
        grid_core["n_regions"] = np.delete(grid_core["n_regions"],-1,1)
        grid_core["n_regions"] = np.delete(grid_core["n_regions"],0,1)
        
        grid_core["s_cap"] = np.delete(grid_core["s_cap"],-1,1)
        grid_core["s_cap"] = np.delete(grid_core["s_cap"],0,1)
        
        """
        data_core["lat"] = np.delete(data_core["lat"],[0,idx_end_core],0)
        data_core["lon"] = np.delete(data_core["lon"],[0,idx_end_core],0)
        data_core["radius"] = np.delete(data_core["radius"],[0,idx_end_core],0)
        data_core["data"] = np.delete(data_core["data"],[0,idx_end_core],0)
        data_core["N"] = idx_end_core-1
        """
        setup_core["N"] = idx_end_core-1
        
        
        
        if grid_core["sph_dist"] is not None:
            grid_core["sph_dist"] = np.delete(grid_core["sph_dist"],[0,idx_end_core],0)
            grid_core["sph_dist"] = np.delete(grid_core["sph_dist"],[0,idx_end_core],1)
    
    if grid_sat is not None:
        idx_end_sat = grid_sat["N"]-1
        grid_sat["lat"] = np.delete(grid_sat["lat"],[0,idx_end_sat],0)
        grid_sat["lon"] = np.delete(grid_sat["lon"],[0,idx_end_sat],0)
        grid_sat["N"] = idx_end_sat-1
        
        """
        data_sat["lat"] = np.delete(data_sat["lat"],[0,idx_end_sat],0)
        data_sat["lon"] = np.delete(data_sat["lon"],[0,idx_end_sat],0)
        data_sat["radius"] = np.delete(data_sat["radius"],[0,idx_end_sat],0)
        data_sat["data"] = np.delete(data_sat["data"],[0,idx_end_sat],0)
        data_sat["N"] = idx_end_sat-1
        """
        setup_sat["N"] = idx_end_sat-1
        
        if grid_sat["sph_dist"] is not None:
            grid_sat["sph_dist"] = np.delete(grid_sat["sph_dist"],[0,idx_end_sat],0)
            grid_sat["sph_dist"] = np.delete(grid_sat["sph_dist"],[0,idx_end_sat],1)
        
        #if np.logical_and(data_core is not None, grid_core is not None):
        if grid_core is not None:
            #return grid_core, data_core, setup_core, grid_sat, data_sat, setup_sat
            return grid_core, setup_core, grid_sat, setup_sat
        #elif np.logical_and(data_sat is None, grid_sat is None):            
        else:
            #return grid_sat, data_sat, setup_sat
            return grid_sat, setup_sat
    else:
        return grid_core, setup_core

def find_sort_d(grid_core, max_dist = 2000):
    import numpy as np
    range_d = grid_core["sph_dist"].ravel() < max_dist
    idx_range = np.array(np.where(range_d == True)).ravel()
    val_range = grid_core["sph_dist"].ravel()[idx_range]
    idx_sort_val_range = np.argsort(val_range)
    sort_d = idx_range[idx_sort_val_range]
    return sort_d

"""
FUNCTIONS RELATED TO GREEN'S
"""

def Gr_vec(r_s, r_d, lat_s, lat_d, lon_s, lon_d, angdist_out = False):
    import numpy as np

    theta_s, theta_d, lon_s, lon_d = map(np.radians, [np.matrix(90.0-lat_s), np.matrix(90.0-lat_d), np.matrix(lon_s), np.matrix(lon_d)])
    
    r_s = np.matrix(r_s)
    r_d = np.matrix(r_d)
    
    mu = np.cos(theta_d.T)*np.cos(theta_s)+np.multiply(np.sin(theta_d.T)
    *np.sin(theta_s),np.cos(lon_d.T-lon_s))
    
    h = r_s.T/r_d
    
    def rs(r_s,r_d, mu):
        r_d_sq = np.power(r_d,2)
        r_s_sq = np.power(r_s,2)
        rr_ds = r_d.T*r_s
        rr_ds_mu = 2*np.multiply(rr_ds,mu)
        rr_ds_sq_sum = r_d_sq.T+r_s_sq
        R = np.sqrt(rr_ds_sq_sum-rr_ds_mu)
        f = R.T/r_d
        return f
    
    f = rs(r_s,r_d, mu)

    h_sq = np.power(h,2)
    
    f_cb = np.power(f,3)
    
    G_r = (1/(4*np.pi)*np.multiply(h_sq,(1-h_sq))/f_cb).T
    if angdist_out == True:
        return G_r, mu
    else:
        return G_r

def take_along_axis(arr, ind, axis):
    import numpy as np
    """
    ... here means a "pack" of dimensions, possibly empty

    arr: array_like of shape (A..., M, B...)
        source array
    ind: array_like of shape (A..., K..., B...)
        indices to take along each 1d slice of `arr`
    axis: int
        index of the axis with dimension M

    out: array_like of shape (A..., K..., B...)
        out[a..., k..., b...] = arr[a..., inds[a..., k..., b...], b...]
    """
    if axis < 0:
       if axis >= -arr.ndim:
           axis += arr.ndim
       else:
           raise IndexError('axis out of range')
    ind_shape = (1,) * ind.ndim
    ins_ndim = ind.ndim - (arr.ndim - 1)   #inserted dimensions

    dest_dims = list(range(axis)) + [None] + list(range(axis+ins_ndim, ind.ndim))

    # could also call np.ix_ here with some dummy arguments, then throw those results away
    inds = []
    for dim, n in zip(dest_dims, arr.shape):
        if dim is None:
            inds.append(ind)
        else:
            ind_shape_dim = ind_shape[:dim] + (-1,) + ind_shape[dim+1:]
            inds.append(np.arange(n).reshape(ind_shape_dim))

    return arr[tuple(inds)]

def greens_differentials(grid):
    import numpy as np
    s_cap = grid["s_cap"].T
    s_cap_diff = np.diff(s_cap,axis=0)

        
    n_regions = grid["n_regions"].T
    #if n_regions[0] == 1:
    s_cap_diff = np.vstack((s_cap[0],s_cap_diff))    
    s_cap_diff[-1] =  s_cap_diff[0]
    d_theta_core = np.empty([0,1],dtype=float)
    d_phi_core = np.empty([0,1],dtype=float)
        
    for i in range(0,len(n_regions)):
            
        d_theta_core = np.vstack((d_theta_core,(s_cap_diff[i]*np.ones(int(n_regions[i]))).T))
            
        d_phi_core = np.vstack((d_phi_core,(2*np.pi/n_regions[i]*np.ones((int(n_regions[i]),1)))))
    
    theta_core = np.matrix(90.0-grid["lat"])*np.pi/180.0
    
    return np.multiply(np.multiply(d_theta_core,d_phi_core),np.sin(theta_core.T))

"""
FUNCTIONS TO CALCULATE EQUAL AREA SPHERICAL COORDINATES

Assumptions:
    - dim is always 2
    - N > 2
"""

#% sradius_of_cap
def sradius_of_cap(area):
    import numpy as np
    #s_cap = 2*np.emath.arcsin(np.sqrt(area/np.pi)/2)
    s_cap = 2*np.arcsin(np.sqrt(area/np.pi)/2)
    return s_cap

#% area_of_sphere
def area_of_sphere():
    import scipy.special as scis
    import numpy as np
    dim = 2
    power = (dim+1)/2
    area = (2*np.pi**power/scis.gamma(power))
    return area

#% area_of_ideal_region
def area_of_ideal_region(N):
    area = area_of_sphere()/N
    return area

#% polar_colat

def polar_colat(N):
    c_polar = sradius_of_cap(area_of_ideal_region(N))
    return c_polar

#% num_collars
def num_collars(N,c_polar,a_ideal):
    import numpy as np
    
    n_collars = np.zeros(np.size(N)).T
    #enough = np.logical_and(N > 2, a_ideal > 0)
    n_collars = max(1,np.round((np.pi-2*c_polar)/a_ideal))
    
    return n_collars

#% ideal_collar_angle
def ideal_collar_angle(N):
    dim = 2
    angle = area_of_ideal_region(N)**(1/dim)
    return angle

#% area_of_cap
def area_of_cap(s_cap):
    import numpy as np
    area = 4*np.pi*np.sin(s_cap/2)**2
    return area

#% area_of_collar
def area_of_collar(a_top, a_bot):
    area = area_of_cap(a_bot) - area_of_cap(a_top);
    return area

#% ideal_region_list
def ideal_region_list(N,c_polar,n_collars):
    import numpy as np
    r_regions = np.zeros((1,2+int(n_collars))).T
    r_regions[0] = 1
    if n_collars > 0:
        a_fitting = (np.pi-2*c_polar)/n_collars
        ideal_region_area = area_of_ideal_region(N)
        for collar_n in range(1,int(n_collars)+1):
            ideal_collar_area = area_of_collar(c_polar+(collar_n-1)*a_fitting, c_polar+collar_n*a_fitting)
            r_regions[0+collar_n] = ideal_collar_area / ideal_region_area
            
    r_regions[1+int(n_collars)] = 1
    
    return r_regions

#% round_to_naturals   
def round_to_naturals(N,r_regions):
    import numpy as np
    n_regions = r_regions
    discrepancy = 0
    for zone_n in range(0,np.size(r_regions,0)):
        n_regions[zone_n] = np.round(r_regions[zone_n]+discrepancy)
        discrepancy = discrepancy+r_regions[zone_n]-n_regions[zone_n]
    
    return n_regions

#% cap_colats
def cap_colats(N,c_polar,n_regions):
    import numpy as np
    c_caps = np.zeros(np.size(n_regions)).T
    c_caps[0] = c_polar
    ideal_region_area = area_of_ideal_region(N)
    n_collars = np.size(n_regions,0)-2
    subtotal_n_regions = 1
    for collar_n in range(1,n_collars+1):
        subtotal_n_regions = subtotal_n_regions+n_regions[0+collar_n]
        c_caps[collar_n+0] = sradius_of_cap(subtotal_n_regions*ideal_region_area)
    
    c_caps[0+n_collars+1] = np.pi
    
    return c_caps

#% eq_caps
def eq_caps(N):
    c_polar = polar_colat(N)

    n_collars = num_collars(N,c_polar,ideal_collar_angle(N))
    
    r_regions = ideal_region_list(N,c_polar,n_collars)
    
    n_regions = round_to_naturals(N,r_regions)
    
    s_cap = cap_colats(N,c_polar,n_regions)
    
    return s_cap, n_regions
    

#% circle_offset   
def circle_offset(n_top,n_bot):
    import numpy as np
    #from math import gcd
    
    offset = (1/n_bot - 1/n_top)/2 + np.gcd(n_top,n_bot)/(2*n_top*n_bot)
    return offset

#% eq_point_set_polar
def eq_point_set_polar(N):
    import numpy as np
    from math import floor
    
    s_cap, n_regions = eq_caps(N)
    
    n_collars = np.size(n_regions,0)-2
    
    points_s = np.zeros((N,2))
    point_n = 1
    offset = 0
    
    cache_size = floor(n_collars/2)
    cache = list()

    for collar_n in range(0,n_collars):
        s_top = s_cap[collar_n]
        s_bot = s_cap[collar_n+1]
        n_in_collar = n_regions[collar_n+1]
        
        twin_collar_n = n_collars-collar_n+1
        
        if (twin_collar_n <= cache_size and np.size(cache[twin_collar_n]) == n_in_collar):
            points_1 = cache[twin_collar_n]
            
        else:
            sector = np.arange(1,n_in_collar+1)
            s_cap_1 = sector*2*np.pi/n_in_collar
            #n_regions_1 = np.ones(len(sector))
            
            points_1 = s_cap_1 - np.pi/n_in_collar
            
            cache.append(points_1)
            
        s_point = (s_top+s_bot)/2
        
        point_1_n = np.arange(0,np.size(points_1))

        #print(point_n+point_1_n)
        points_s[point_n+point_1_n,0] = (points_1[point_1_n]+2*np.pi*offset)%(2*np.pi)

        offset = offset + circle_offset(int(n_in_collar),int(n_regions[2+collar_n]))
        offset = offset - floor(offset)

        points_s[point_n+point_1_n,1] = s_point
        point_n = point_n + np.size(points_1)
    
    points_s[point_n,:] = np.zeros((1,2))
    points_s[point_n,1] = np.pi
    
    return points_s