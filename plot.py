"""
Plotting for SDSSIM
"""

def plots(plot_type, ipt, *args, bins=50, setup = None, save_path = None, prior_dipole = False, point_size = 10):
    import matplotlib.pyplot as plt
    
    if plot_type == 'target histogram':
        plt.figure(figsize=setup["figsize"])
        plt.hist(ipt["data"],bins=200)
        plt.title('Target histogram for %s model' % setup["data_type"], fontsize=setup["fontsize"])
        plt.xticks(fontsize=setup["gensize"])
        plt.yticks(fontsize=setup["gensize"])
        plt.xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        if setup["savefig"] == True:
            if save_path == None:
                return
            else:
                plt.savefig('%shist_target_%s.png' % (save_path,setup["data_type"]), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
        
    elif plot_type == 'grid':
        from SDSSIM_utility import basemap_plot
        basemap_plot(ipt["grid latitude"], ipt["grid longitude"], 0.0, 0.0, None, data=None, figsize=setup["figsize"], title='EQSP grid', titlefontsize=setup["fontsize"], projection='ortho', lat_0 = 45.0, lon_0 = 0.0, point_size=3, savefig = setup["savefig"], dpi = setup["dpi"], path = '%sgrid.png' %save_path)
    
    elif plot_type == 'data_basemap':
        from SDSSIM_utility import basemap_plot
        #basemap_plot(args[0], args[1], min(ipt["data"]), max(ipt["data"]), setup["unit"], data=ipt["data"], cmap = setup["cmap"], figsize=setup["figsize"], title='%s' % setup["type"], titlefontsize=setup["fontsize"], projection='hammer', lat_0 = 0.0, lon_0 = 0.0, point_size=point_size, savefig = setup["savefig"], path = '%s%s.png' % (save_path,setup["type_model"]))
        basemap_plot(args[0], args[1], min(ipt["data"]), max(ipt["data"]), setup["unit"], data=ipt["data"], cmap = setup["cmap"], figsize=setup["figsize"], title=args[2], titlefontsize=setup["fontsize"], projection='hammer', lat_0 = 0.0, lon_0 = 0.0, point_size=point_size, savefig = setup["savefig"], path = '%s%s.png' % (save_path,plot_type))

    elif plot_type == 'data_basemap_adjust':
        from SDSSIM_utility import basemap_plot
        basemap_plot(args[0], args[1], min(args[2]["data"]), max(args[2]["data"]), setup["unit"], data=ipt["data"], cmap = setup["cmap"], figsize=setup["figsize"], title='Radial field', titlefontsize=setup["fontsize"], projection='hammer', lat_0 = 0.0, lon_0 = 0.0, point_size=point_size, savefig = setup["savefig"], path = '%s%s.png' % (save_path,setup["type_model"]))
        
    elif plot_type == 'target_semi_variogram':
        plt.figure(figsize=setup["figsize"])
        plt.plot(ipt["total data lags"],ipt["total data sv"],'o', markersize=10)
        #plt.title('Semi-variogram of target distribution',fontsize=setup["fontsize"],y=1.02)
        plt.ylabel('Semi-variance $%s^2$' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.xlabel('Lag [km]',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = setup["gensize"])
        if setup["savefig"] == True:
            plt.savefig('%starget_sv_%s.png' % (save_path, ipt["sv model"]), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
    
    elif plot_type == 'model_semi_variogram':
        plt.figure(figsize=setup["figsize"]) 
        plt.plot(ipt["total data lags"],ipt["total data sv"],'o', markersize=10,color = '0.85',label='data semi-variogram')
        plt.plot(ipt["model data lags"],ipt["model data sv"],'.',markersize=10,label='modelling data')
        plt.plot(ipt["sv model x"],ipt["sv model y"],color='C1',linewidth = 3,label='model')
        plt.ylabel('Semi-variance $%s^2$' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.xlabel('Lag [km]',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.title("Semi-variogram model of type: %s " % ipt["model names"][ipt["sv model"]],fontsize=setup["fontsize"],y=1.02)
        plt.legend(loc='best',fontsize=setup["fontsize"])
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = setup["gensize"])
        if setup["savefig"] == True:
            plt.savefig('%smodel_sv_%s.png' % (save_path, ipt["sv model"]), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()  
        
    elif plot_type == 'model_semi_variogram_new':
        plt.figure(figsize=(10,10)) 
        plt.plot(ipt["total data lags"],ipt["total data sv"],'o', markersize=10,color = '0.85',label='data semi-variogram')
        plt.plot(ipt["model data lags"],ipt["model data sv"],'.',markersize=10,label='modelling data')
        plt.plot(ipt["sv model x"],ipt["sv model y"],color='C1',linewidth = 3,label='model')
        plt.ylabel('Semi-variance $[%s^2]$' % "nT",fontsize=18,labelpad=18)
        plt.xlabel('Lag [km]',fontsize=18,labelpad=18)
        plt.title("Semi-variogram model of type: %s " % ipt["model names"][ipt["sv model"]],fontsize=18,y=1.02)
        plt.legend(loc='best',fontsize=18)
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
        plt.show()  
        
    elif plot_type == 'semi-variogram LUT':
        plt.figure(figsize=setup["figsize"])
        plt.imshow(ipt["semi-variogram LUT"][:5000,:5000])
        cbar = plt.colorbar(shrink=0.8, pad=0.02)
        cbar.ax.set_ylabel('Semi-variance $%s^2$' % setup["unit"], fontsize=setup["fontsize"], labelpad=setup["gensize"])
        plt.xlabel('LUT column number',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('LUT row number',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.title('LUT for %s field with sv-model: %s' % (setup["type_model"], ipt["model names"][ipt["sv model"]]), fontsize=setup["fontsize"], y=1.02)
        if setup["savefig"] == True:
            plt.savefig('%sLUT_%s_%s.png' % (save_path, setup["type_model"], ipt["sv model"]), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()        
    
    elif plot_type == 'data vs normal QFs':
        import numpy as np
        from scipy.stats import norm
        
        linsize = 1000
        linspace = np.linspace(args[0],1-args[0],linsize)
        target_mean = np.mean(ipt["data"])
        target_var = np.var(ipt["data"])
        data_sorted = np.sort(ipt["data"])
        lin_ppf = norm.ppf(linspace,loc=target_mean,scale=np.sqrt(target_var))
        plt.figure(figsize=setup["figsize"]) 
        plt.plot(np.linspace(0,1,linsize),lin_ppf,linewidth = 2,label='Quantile function at data mean and variance')
        plt.plot(np.linspace(0,1,setup["N"]),data_sorted,linewidth = 2,label='Data quantile function')
        plt.legend(loc='lower right',prop={'size': setup["gensize"]})
        plt.title('Quantile functions of %s model and corresponding normal distribution' % setup["type_model"], fontsize=setup["fontsize"], y=1.02)
        plt.xticks(fontsize=setup["gensize"])
        plt.yticks(fontsize=setup["gensize"])
        plt.xlabel('$P(Z\leq F)$', fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('$Z_F$', fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.grid()
        if setup["savefig"] == True:
            plt.savefig('%squanfunc_%s.png' % (save_path, setup["data_type"]), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()  
    
    elif plot_type == 'data normal score transform':
        
        plt.figure(figsize=setup["figsize"])
        plt.hist(ipt["target normscore"],bins=200)
        plt.title('Normal score transformation of %s model (%s)' % (setup["type_model"], ipt["compiler"]), fontsize=setup["fontsize"])
        plt.xticks(fontsize=setup["gensize"])
        plt.yticks(fontsize=setup["gensize"])
        plt.xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('Count',fontsize=setup["fontsize"], labelpad=setup["gensize"])
        if setup["savefig"] == True:
            plt.savefig('%sns_%s_%s.png' % (save_path, setup["type_model"], ipt["compiler"]), bbox_inches='tight', dpi = setup["dpi"])
    
    elif plot_type == 'normal QF range':
        from matplotlib.lines import Line2D
        from SDSSIM_utility import printProgressBar
        import numpy as np
        from scipy.stats import norm
        
        linspace = np.linspace(ipt["start"],1-ipt["start"],ipt["normsize"])
        custom_lines = [Line2D([0], [0], color=setup["cmap"](0.1), lw=2), Line2D([0], [0], color=setup["cmap"](0.5), lw=2), Line2D([0], [0], color=setup["cmap"](0.9), lw=2)]
        len_QFN = len(ipt["QF norm range"])
        len_QFV = len(ipt["QF var range"])
        plt.figure(figsize=setup["figsize"]) 
        for i in range(0,len_QFN,1):
            printProgressBar (i, len_QFN, subject = 'done with normal quantile plot')
            j = int(len_QFV/2)
            lin_ppf = norm.ppf(linspace,loc=ipt["QF norm range"][i],scale=np.sqrt(ipt["QF var range"][j]))
            plt.plot(np.linspace(0,1,ipt["normsize"]),lin_ppf,'-',linewidth = 2, color = setup["cmap"](i/len_QFN))
                
        plt.xticks(fontsize=setup["gensize"])
        plt.yticks(fontsize=setup["gensize"])
        plt.title('Normal quantile functions to be back-transformed',fontsize=setup["fontsize"])
        plt.xlabel('$P(Z\leq F)$', fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('$Z_F$', fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.legend(custom_lines, ['Negative', 'Mean', 'Positive'],fontsize=setup["fontsize"])
        plt.ylim(-5,5)
        if setup["savefig"] == True:
            plt.savefig('%squanfunc_back.png' % save_path, bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
        
    elif plot_type == 'CQF range':
        from matplotlib.lines import Line2D
        from SDSSIM_utility import printProgressBar
        import numpy as np
        
        linspace = np.linspace(ipt["start"],1-ipt["start"],ipt["normsize"])
        custom_lines = [Line2D([0], [0], color=setup["cmap"](0.1), lw=2), Line2D([0], [0], color=setup["cmap"](0.5), lw=2), Line2D([0], [0], color=setup["cmap"](0.9), lw=2)]
        len_QFN = len(ipt["QF norm range"])
        len_QFV = len(ipt["QF var range"])
        plt.figure(figsize=setup["figsize"]) 
        for i in range(0,len_QFN,1):
            printProgressBar (i, len_QFN, subject = 'done with back-transformation plot')
            j = int(len_QFV/2)
            plt.plot(np.linspace(0,1,ipt["normsize"]),ipt["CQF dist"][i,j,:],'-',linewidth = 2,color = setup["cmap"](i/len_QFN))
                
        plt.xticks(fontsize=setup["gensize"])
        plt.yticks(fontsize=setup["gensize"])
        plt.legend(loc='lower right',prop={'size': setup["gensize"]})
        plt.title('Quantile functions after back-transformation for %s model' % setup["type_model"],fontsize= setup["fontsize"])
        plt.xlabel('$P(Z\leq F)$', fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('$Z_F$', fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.legend(custom_lines, ['Negative', 'Mean', 'Positive'],fontsize=setup["fontsize"])
        if setup["savefig"] == True:    
            plt.savefig('%squanfunc_backed_%s_%s_%d.png' % (save_path, setup["data_type"], ipt["compiler"], ipt["normsize"]), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
        
    elif plot_type == 'mean and var CQF coverage':
        plt.figure(figsize=setup["figsize"])
        plt.plot(ipt["CQF mean"],ipt["CQF var"],'C0.')
        plt.ylabel('Variance $%s^2$' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.xlabel('Mean %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.title('Mean and variance of %s model conditional distributions' % setup["type_model"],fontsize=setup["fontsize"])
        plt.xticks(fontsize=setup["gensize"])
        plt.yticks(fontsize=setup["gensize"])
        if setup["savefig"] == True:    
            plt.savefig('%sEV_%s_%s_%d.png' % (save_path, setup["data_type"], ipt["compiler"], ipt["normsize"]), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
    
    elif plot_type == 'realization_basemap':
        from SDSSIM_utility import basemap_plot
        import numpy as np
        for run_plot in range(0,min(5,ipt["realization amount"])):
            basemap_plot(args[0], args[1], min(args[2]), max(args[2]), setup["unit"], data=ipt["realizations"][:,run_plot], cmap = setup["cmap"], figsize=setup["figsize"], title='SDSSIM of %s, realization nr. %d %s' % (setup["type_model"],run_plot+1, setup["unit"]), titlefontsize=setup["fontsize"], projection='hammer', lat_0 = 0.0, lon_0 = 0.0, point_size=point_size, savefig = setup["savefig"], path = '%srealization_run%d' % (save_path,run_plot+1))
            if np.logical_and(ipt["return Julien mean"] == 'on', setup["data_type"] == 'Julien'):
               basemap_plot(args[0], args[1], min(args[2]), max(args[2]), setup["unit"], data=ipt["realizations with mean"][:,run_plot], cmap = setup["cmap"], figsize=setup["figsize"], title='SDSSIM of %s with added mean, realization nr. %d %s' % (setup["type_model"], run_plot+1, setup["unit"]), titlefontsize=setup["fontsize"], projection='hammer', lat_0 = 0.0, lon_0 = 0.0, point_size=point_size, savefig = setup["savefig"], path = '%srealization_meanreturn_run%d' % (save_path,run_plot+1))
        basemap_plot(args[0], args[1], min(args[2]), max(args[2]), setup["unit"], data=np.mean(ipt["realizations"],axis=1), cmap = setup["cmap"], figsize=setup["figsize"], title='Mean of realizations for %s field %s' % (setup["type_model"], setup["unit"]), titlefontsize=setup["fontsize"], projection='hammer', lat_0 = 0.0, lon_0 = 0.0, point_size=point_size, savefig = setup["savefig"], path = '%srealization_mean' % save_path)
            
    elif plot_type == 'realization_histogram':
        import numpy as np
        plt.figure(figsize=setup["figsize"])
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["realizations"][:,j],bins=200)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                plt.plot(bincenters,y,'-',color = '0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["realizations"][:,j+1],bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'-',color = '0.75',label='Realizations')    
        
        y,binEdges=np.histogram(args[0],bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'k-',label='Target')
        #plt.title("Distribution of all realized priors and the target",fontsize=setup["fontsize"])
        plt.legend(loc='best',fontsize=setup["fontsize"])
        plt.xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        if setup["savefig"] == True: 
            plt.savefig('%s%s.png' % (save_path, plot_type), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
        
    elif plot_type == 'prediction_histogram':
        import numpy as np
        plt.figure(figsize=setup["figsize"])
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["data prediction"][:,j],bins=200)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                plt.plot(bincenters,y,'-',color = '0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["data prediction"][:,j+1],bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'-',color = '0.75',label='Predictions')    
        
        y,binEdges=np.histogram(args[0],bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'k-',label='Target')
        #plt.title("Distribution of all realized priors and the target",fontsize=setup["fontsize"])
        plt.legend(loc='best',fontsize=setup["fontsize"])
        plt.xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        if setup["savefig"] == True: 
            plt.savefig('%s%s.png' % (save_path, plot_type), bbox_inches='tight', dpi = setup["dpi"])
        plt.show() 
        
    elif plot_type == 'realization_semi_variogram':
        from SDSSIM_semivar import sv_sim_cloud
        from SDSSIM_utility import variable_load
        import numpy as np
        l_var = 1
        lag_coarse = int(ipt["n_lags"]/l_var)
        cloud_coarse = int(ipt["max_cloud"]/lag_coarse)
        
        if setup["data_type"] == "Julien small":
            sph_d_sorted = ipt["sph_d_sorted"]
        else:
            sph_d_sorted = variable_load('saved_variables/sph_d_ravel_short_sort.npy')
            sph_d_sorted = np.multiply(setup["grid_radius"],sph_d_sorted)
        
        
        cloud_zs = sv_sim_cloud(lag_coarse, cloud_coarse, args[0]["realization amount"], args[0]["realizations"], setup["N"], sort_d = ipt["sort_d"], data_type = setup["data_type"])
        lags_posterior = np.array([np.mean(sph_d_sorted[n*cloud_coarse:cloud_coarse*(n+1)]) for n in range(0,lag_coarse)])
        
        plt.figure(figsize=setup["figsize"])
        if args[0]["realization amount"]>1:
            for j in range(0,args[0]["realization amount"]-1):
                plt.plot(lags_posterior,cloud_zs[:,j],color = '0.75')
        else:
            j = -1
            
        plt.plot(lags_posterior,cloud_zs[:,j+1],color = '0.75',label='Realizations')
        plt.plot(lags_posterior,np.mean(cloud_zs, axis=1),'k-',label='Realization mean')
        plt.plot(ipt["model data lags"],ipt["model data sv"],'.',markersize=10,label='modelling data')
        plt.plot(ipt["sv model x"],ipt["sv model y"],color='C1', label='%s model' % ipt["model names"][ipt["sv model"]])
        plt.ylabel('Semi-variance $%s^2$' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.xlabel('Lag [km]',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        #plt.title("Semi-variogram of realizations compared to model",fontsize=setup["fontsize"])
        plt.legend(loc='best',fontsize=setup["fontsize"])
        if setup["savefig"] == True: 
            plt.savefig('%s%s.png' % (save_path, plot_type), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
        
    elif plot_type == 'realization_semi_variogram_pc':
        plt.figure(figsize=setup["figsize"])
        if ipt["realization amount"]>1:
            for j in range(0,ipt["realization amount"]-1):
                plt.plot(ipt["realizations sv lags"],ipt["realizations sv"][:,j],color = '0.75')
        else:
            j = -1
            
        plt.plot(ipt["realizations sv lags"],ipt["realizations sv"][:,j+1],color = '0.75',label='Realizations')
        
        plt.plot(args[0]["model data lags"],args[0]["model data sv"],'.',markersize=10,label='Semi-variogram modelling data')
        plt.plot(ipt["realizations sv lags"],ipt["realizations mean sv"],'k-',label='Realization mean')
        plt.plot(args[0]["sv model x"],args[0]["sv model y"],color='C1', linewidth = 3, label='%s model' % args[0]["model names"][args[0]["sv model"]])
        plt.ylabel('Semi-variance $%s^2$' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.xlabel('Lag [km]',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        #plt.title("Semi-variogram of realizations compared to model",fontsize=setup["fontsize"])
        plt.legend(loc='best',fontsize=setup["fontsize"])
        if setup["savefig"] == True: 
            plt.savefig('%s%s.png' % (save_path, plot_type), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
        
    elif plot_type == 'data_misfit':
        import numpy as np
        plt.figure(figsize=setup["figsize"])
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["misfit_forward"][:,j],bins=args[0])
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                plt.plot(bincenters,y,'-',color = '0.75')
                #plt.hist(ipt["misfit_forward"][:,j],bins=args[0], color='0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["misfit_forward"][:,j+1],bins=args[0])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'-',color = '0.75',label='Data realization misfit')    
        #plt.hist(ipt["misfit_forward"][:,j+1],bins=args[0], color='0.75', label='Misfit of forward computed realizations')
        
        y,binEdges=np.histogram(ipt["misfit_forward_mean"],bins=args[0])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'k-',label='Misfit for mean of forward computed realizations')
        #plt.title("Data Misfit",fontsize=setup["fontsize"])
        plt.legend(loc='best',fontsize=setup["fontsize"])
        plt.xlabel('Data misfit, $B(\mathbf{r})_{est} - B(\mathbf{r})_{obs}$  %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        if setup["savefig"] == True: 
            plt.savefig('%s%s.png' % (save_path, plot_type), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
        
    elif plot_type == 'prior_misfit':
        import numpy as np
        plt.figure(figsize=setup["figsize"])
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["misfit_prior"][:,j],bins=args[0])
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                plt.plot(bincenters,y,'-',color = '0.75')
                #plt.hist(ipt["misfit_prior"][:,j],bins=args[0], color='0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["misfit_prior"][:,j+1],bins=args[0])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'-',color = '0.75',label='Misfit of realizations')    
        #plt.hist(ipt["misfit_prior"][:,j+1],bins=args[0], color='0.75', label='Misfit of realizations')
        
        y,binEdges=np.histogram(ipt["misfit_prior_mean"],bins=args[0])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'k-',label='Misfit for mean of realizations')
        #plt.title("Prior Misfit",fontsize=setup["fontsize"])
        plt.legend(loc='best',fontsize=setup["fontsize"])
        plt.xlabel("Prior misfit, $B_r(\mathbf{r'})_{est} - B_r(\mathbf{r'})_{prior}$ %s" % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        if setup["savefig"] == True: 
            plt.savefig('%s%s.png' % (save_path, plot_type), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
        
    elif plot_type == 'fit_diag':
        import numpy as np
        from matplotlib.font_manager import FontProperties
        fig, axes = plt.subplots(3, 2, figsize=(20,20))
        font=FontProperties()
        font.set_size('x-large')
        
        # DATA HISTOGRAM FIT
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["data prediction"][:,j],bins=200)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                axes[0,0].plot(bincenters,y,'-',color = '0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["data prediction"][:,j+1],bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,0].plot(bincenters,y,'-',color = '0.75',label='Predictions')    
        
        y,binEdges=np.histogram(args[3],bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,0].plot(bincenters,y,'k-',label='Target')
        
        y,binEdges=np.histogram(np.mean(ipt["data prediction"],axis=1),bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,0].plot(bincenters,y,'r--',label='Mean', linewidth=1)
        
        axes[0,0].legend(loc='upper center',fontsize=setup["fontsize_sub"])
        axes[0,0].set_xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        axes[0,0].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        
        stddev_d = np.std(args[3])
        var_d = np.var(args[3])
        mean_d = np.mean(args[3])
        #stattext = '\\textbf{Data}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
        #axes[0,0].text(-15000, 160, stattext, fontproperties=font)
        
        stddev = np.std(ipt["data prediction"])
        var = np.var(ipt["data prediction"])
        mean = np.mean(ipt["data prediction"])
        
        minx = 1.0*np.min(bincenters)
        maxy = 0.65*np.max(y)
        
        #stattext = '\\textbf{Data prediction mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
        
        stattext = '\\textbf{Observations}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Prediction mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_d, stddev_d, var_d, mean, stddev, var)
        axes[0,0].text(minx, maxy, stattext, fontproperties=font)
        
        # PRIOR HISTOGRAM FIT
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["realizations"][:,j],bins=bins)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                axes[0,1].plot(bincenters,y,'-',color = '0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["realizations"][:,j+1],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,1].plot(bincenters,y,'-',color = '0.75',label='Realizations of prior')    

        y,binEdges=np.histogram(args[5],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,1].plot(bincenters,y,'k--',label='Prior w/o dipole')
        
        y,binEdges=np.histogram(args[0],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,1].plot(bincenters,y,'k-',label='Prior w/ dipole')
        axes[0,1].legend(loc='upper right',fontsize=setup["fontsize_sub"])
        axes[0,1].set_xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        axes[0,1].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        
        stddev_p = np.std(args[0])
        var_p = np.var(args[0])
        mean_p = np.mean(args[0])
        #stattext = '\\textbf{Prior}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
        #axes[0,1].text(2000000, 350, stattext, fontproperties=font)
        
        stddev = np.std(ipt["realizations"])
        var = np.var(ipt["realizations"])
        mean = np.mean(ipt["realizations"])
        
        minx = 1.0*np.min(bincenters)
        maxy = 0.6*np.max(y)
        
        stattext = '\\textbf{Prior}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Realization mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_p, stddev_p, var_p, mean, stddev, var)
        axes[0,1].text(minx, maxy, stattext, fontproperties=font)

    elif plot_type == 'fit_diag_new':
        import numpy as np
        from matplotlib.font_manager import FontProperties
        fig, axes = plt.subplots(3, 2, figsize=(20,20))
        font=FontProperties()
        font.set_size('x-large')
        
        # DATA HISTOGRAM FIT
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["data prediction"][:,j],bins=200)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                axes[0,0].plot(bincenters,y,'-',color = '0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["data prediction"][:,j+1],bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,0].plot(bincenters,y,'-',color = '0.75',label='Predictions')    
        
        y,binEdges=np.histogram(args[3],bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,0].plot(bincenters,y,'k-',label='Target')
        
        y,binEdges=np.histogram(np.mean(ipt["data prediction"],axis=1),bins=200)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,0].plot(bincenters,y,'r--',label='Mean', linewidth=1)
        
        axes[0,0].legend(loc='upper center',fontsize=setup["fontsize_sub"])
        axes[0,0].set_xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        axes[0,0].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        
        stddev_d = np.std(args[3])
        var_d = np.var(args[3])
        mean_d = np.mean(args[3])
        #stattext = '\\textbf{Data}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
        #axes[0,0].text(-15000, 160, stattext, fontproperties=font)
        
        stddev = np.std(ipt["data prediction"])
        var = np.var(ipt["data prediction"])
        mean = np.mean(ipt["data prediction"])
        
        minx = 1.0*np.min(bincenters)
        maxy = 0.65*np.max(y)
        
        #stattext = '\\textbf{Data prediction mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
        
        stattext = '\\textbf{Observations}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Prediction mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_d, stddev_d, var_d, mean, stddev, var)
        axes[0,0].text(minx, maxy, stattext, fontproperties=font)
        
        # PRIOR HISTOGRAM FIT
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["realizations"][:,j],bins=bins)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                axes[0,1].plot(bincenters,y,'-',color = '0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["realizations"][:,j+1],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,1].plot(bincenters,y,'-',color = '0.75',label='Realizations of prior')    

        #y,binEdges=np.histogram(args[5],bins=bins)
        #bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        #axes[0,1].plot(bincenters,y,'k--',label='Prior w/o dipole')
        
        y,binEdges=np.histogram(args[0],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[0,1].plot(bincenters,y,'k-',label='Prior w/ dipole')
        axes[0,1].legend(loc='upper right',fontsize=setup["fontsize_sub"])
        axes[0,1].set_xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        axes[0,1].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        
        stddev_p = np.std(args[0])
        var_p = np.var(args[0])
        mean_p = np.mean(args[0])
        #stattext = '\\textbf{Prior}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
        #axes[0,1].text(2000000, 350, stattext, fontproperties=font)
        
        stddev = np.std(ipt["realizations"])
        var = np.var(ipt["realizations"])
        mean = np.mean(ipt["realizations"])
        
        minx = 1.0*np.min(bincenters)
        maxy = 0.6*np.max(y)
        
        stattext = '\\textbf{Prior}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Realization mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_p, stddev_p, var_p, mean, stddev, var)
        axes[0,1].text(minx, maxy, stattext, fontproperties=font)
        
        # INTEGRATION RESIDUALS
        
        if args[4] is "idx_plot":
            import statsmodels.stats.api as sms
            
            conf_int = sms.DescrStatsW(ipt["data prediction"].T).tconfint_mean()
            
            idx_sort = args[3].argsort()
            
            if ipt["realization amount"] > 1:
                for j in range(0,ipt["realization amount"]-1):
                    axes[1,0].plot(ipt["data prediction"][idx_sort,j],'-',color = '0.75')
                    #axes[1,0].semilogy(abs(ipt["data prediction"][idx_sort,j]),'-',color = '0.75')
            else:
                j = -1
                
            axes[1,0].plot(ipt["data prediction"][idx_sort,j+1],'-',color = '0.75',label='Predictions')    
            #axes[1,0].semilogy(abs(ipt["data prediction"][idx_sort,j+1]),'-',color = '0.75',label='Predictions')    
            
            axes[1,0].plot(conf_int[0][idx_sort],'r--',label='$95\%$ CI')
            axes[1,0].plot(conf_int[1][idx_sort],'r--')
            
            axes[1,0].plot(args[3][idx_sort],'k-',label='Target')
            
            #axes[1,0].semilogy(abs(args[3][idx_sort]),'k-',label='Target')
            
            #axes[1,0].plot(np.mean(ipt["data prediction"],axis=1)[idx_sort],'r--',label='Mean')
            
            axes[1,0].legend(loc='upper left',fontsize=setup["fontsize_sub"])
            axes[1,0].set_xlabel("sorted observation index",fontsize=setup["fontsize"],labelpad=setup["gensize"])
            axes[1,0].set_ylabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
            
            #stddev_d = np.std(args[3])
            #var_d = np.var(args[3])
            #mean_d = np.mean(args[3])
            #stattext = '\\textbf{Data}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
            #axes[0,0].text(-15000, 160, stattext, fontproperties=font)
            
            #stddev = np.std(ipt["data prediction"])
            #var = np.var(ipt["data prediction"])
            #mean = np.mean(ipt["data prediction"])
            
            #minx = np.min(bincenters)
            #maxy = 0.7*np.max(y)
            
            #stattext = '\\textbf{Data prediction mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
            
            #stattext = '\\textbf{Data}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Data prediction mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_d, stddev_d, var_d, mean, stddev, var)
            #axes[0,0].text(minx, maxy, stattext, fontproperties=font)
        elif args[4] is "rms_misfit":
            rms_misfit = np.sqrt(np.sum(np.power(ipt["residual_forward"],2),axis=1)/ipt["residual_forward"].shape[1])
            
            #ssqres_stat = np.sum(np.power(ipt["data prediction"]-np.mean(ipt["data prediction"],axis=1),2),axis=1)/np.var(args[3])
            #ssqres_stat = np.sum(np.power(args[3]-np.mean(ipt["data prediction"],axis=1),2),axis=1)/np.var(args[3])
            
            #idx_sort = args[3].argsort()
            
            y,binEdges=np.histogram(rms_misfit,bins=bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[1,0].plot(bincenters,y,'k-',label='$\sqrt{\sum_{i=1}^{N_{sim}}(B(\mathbf{r})_{obs} - B(\mathbf{r})_{est})_i^2/N_{sim}}$')  
            #axes[1,0].plot(rms_misfit[idx_sort],'k-',label='$\sqrt{\sum_{i=1}^{N_{sim}}(B(\mathbf{r})_{obs} - B(\mathbf{r})_{est})^2/N_{sim}}$')  
            axes[1,0].set_xlabel("Observation prediction RMS misfit %s" %setup["unit"], fontsize=setup["fontsize"],labelpad=setup["gensize"])
            #axes[1,0].set_ylabel("RMS misfit %s" %setup["unit"], fontsize=setup["fontsize"],labelpad=setup["gensize"])
            axes[1,0].legend(loc='upper right',fontsize=setup["fontsize_sub"])
            axes[1,0].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
            #axes[1,0].set_xlabel('sorted observation index',fontsize=setup["fontsize"],labelpad=setup["gensize"])                   
            
        elif args[4] is not None:
            test_pred_prior = args[4]*np.matrix(args[0]).T
            test_residual = np.matrix(args[3]).T-test_pred_prior
            
            y,binEdges=np.histogram(test_residual,bins=bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[1,0].plot(bincenters,y,'k-',label='Integration residuals')  
            axes[1,0].set_xlabel("Integration residual, $B(\mathbf{r})_{obs} - B(\mathbf{r})_{prior}$ %s" %setup["unit"], fontsize=setup["fontsize"],labelpad=setup["gensize"])
            axes[1,0].legend(loc='upper left',fontsize=setup["fontsize_sub"])
            axes[1,0].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])        
            stddev = np.std(test_residual)
            var = np.var(test_residual)
            mean = np.mean(test_residual)
            
            minx = np.min(bincenters)
            maxy = 0.5*np.max(y)
            
            stattext = '\\textbf{Statistics}\n$\mu=%.3g$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
            axes[1,0].text(minx, maxy, stattext, fontproperties=font)
        else:
            y,binEdges=np.histogram(ipt["kriging_mv"][0][:,0],bins=bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[1,0].plot(bincenters,y,'k-')  
            
            #axes[1,0].hist(ipt["kriging_mv"][0][:,0],bins=31)
            axes[1,0].set_xlabel("Mean of sampled local conditional distributions %s" %setup["unit"], fontsize=setup["fontsize"],labelpad=setup["gensize"])
            axes[1,0].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])  
            
        # SEMI-VARIOGRAM FIT
        if ipt["realization amount"]>1:
            for j in range(0,ipt["realization amount"]-1):
                axes[1,1].plot(ipt["realizations sv lags"],ipt["realizations sv"][:,j],color = '0.75')
        else:
            j = -1
        axes[1,1].plot(ipt["realizations sv lags"],ipt["realizations sv"][:,j+1],color = '0.75',label='Realizations')
        if prior_dipole == True:
            axes[1,1].plot(args[2]["total data lags"],args[2]["total data sv"],'.', color='C2',markersize=2,label='Target semi-variogram')
        axes[1,1].plot(args[1]["total data lags"],args[1]["total data sv"],'.',markersize=2,label='Modelled semi-variogram')
        axes[1,1].plot(ipt["realizations sv lags"],ipt["realizations mean sv"],'k-',label='Realization mean')
        axes[1,1].plot(args[1]["sv model x"],args[1]["sv model y"],color='C1', linewidth = 2, label='%s model' % args[1]["model names"][args[1]["sv model"]])
        axes[1,1].set_ylabel('Semi-variance $%s^2$' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        axes[1,1].set_xlabel('Lag [km]',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        axes[1,1].legend(loc='lower right', ncol=2, fontsize=setup["fontsize_sub"], markerscale=4)  
        
        # DATA RESIDUAL
        axes[2,0].set_xlabel('Observation prediction residuals, $B(\mathbf{r})_{obs} - B(\mathbf{r})_{est}$  %s' %setup["unit"], fontsize=setup["fontsize"],labelpad=setup["gensize"])

        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["residual_forward"][:,j],bins=bins)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                axes[2,0].plot(bincenters,y,'-',color = '0.75')
        else:
            j = -1
        
        #maxy = np.max(y)
        
        y,binEdges=np.histogram(ipt["residual_forward"][:,j+1],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[2,0].plot(bincenters,y,'-',color = '0.75',label='Prediction residuals')
        
        minx = np.min(ipt["residual_forward"])
        maxy = 4.5*np.mean(y)
        
        y,binEdges=np.histogram(ipt["residual_forward_mean"],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[2,0].plot(bincenters,y,'k-',label='Mean residual')
        axes[2,0].legend(loc='upper right',fontsize=setup["fontsize_sub"])
        axes[2,0].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        stddev = np.std(ipt["residual_forward"])
        var = np.var(ipt["residual_forward"])
        mean = np.mean(ipt["residual_forward"])
        stddev_m = np.std(ipt["residual_forward_mean"])
        var_m = np.var(ipt["residual_forward_mean"])
        mean_m = np.mean(ipt["residual_forward_mean"])   
        
        stattext = '\\textbf{Mean realization statistics}\n$\mu=%.3g$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Mean statistics}\n$\mu=%.3g$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var,mean_m, stddev_m, var_m)
        axes[2,0].text(minx, maxy, stattext, fontproperties=font)
        
        # PRIOR RESIDUAL
        if ipt["realization amount"] > 1:
            for j in range(0,ipt["realization amount"]-1):
                y,binEdges=np.histogram(ipt["residual_prior"][:,j],bins=bins)
                bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                axes[2,1].plot(bincenters,y,'-',color = '0.75')
        else:
            j = -1
            
        y,binEdges=np.histogram(ipt["residual_prior"][:,j+1],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[2,1].plot(bincenters,y,'-',color = '0.75',label='Realization residuals')    
        
        minx = np.min(ipt["residual_prior"])
        maxy = 0.5*np.max(y)
        
        y,binEdges=np.histogram(ipt["residual_prior_mean"],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        axes[2,1].plot(bincenters,y,'k-',label='Mean residual')
        
        
        axes[2,1].legend(loc='upper right',fontsize=setup["fontsize_sub"])
        axes[2,1].set_xlabel("Prior residuals, $B(\mathbf{r'})_{prior} - B(\mathbf{r'})_{est}$ %s" %setup["unit"], fontsize=setup["fontsize"],labelpad=setup["gensize"])
        axes[2,1].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        stddev = np.std(ipt["residual_prior"])
        var = np.var(ipt["residual_prior"])
        mean = np.mean(ipt["residual_prior"])
        stddev_m = np.std(ipt["residual_prior_mean"])
        var_m = np.var(ipt["residual_prior_mean"])
        mean_m = np.mean(ipt["residual_prior_mean"])        

        
        stattext = '\\textbf{Mean realization statistics}\n$\mu=%.3g$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Mean statistics}\n$\mu=%.3g$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var,mean_m, stddev_m, var_m)
        axes[2,1].text(minx, maxy, stattext, fontproperties=font)
        
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[5]), bbox_inches='tight', dpi = setup["dpi"])
        
    elif plot_type == 'neighborhood_size':
        from SDSSIM_utility import haversine
        import numpy as np
        simspace_diag = haversine(setup["grid_radius"], ipt["simspaces"][0][:,0], ipt["simspaces"][0][:,2], ipt["simspaces"][0][:,1], ipt["simspaces"][0][:,3])/(2*np.pi*setup["grid_radius"])*100
        sq_geodiff = np.abs(ipt["simspaces"][0][:,0]-ipt["simspaces"][0][:,1])*np.abs(ipt["simspaces"][0][:,2]-ipt["simspaces"][0][:,3])
        
        idx_sort = simspace_diag.argsort()
        idx_sort_back = idx_sort.argsort()
        
        cline = np.linspace(0,1,len(idx_sort))
        
        fig, axes = plt.subplots(1, 2, figsize=setup["figsize"])
        axes[0].scatter(ipt["simspaces"][0][:,4],simspace_diag, s=1, c=cline[idx_sort_back], cmap=plt.cm.Spectral_r)
        axes[0].set_xlabel("Step in random path",fontsize=setup["fontsize"])
        axes[0].set_ylabel("Neighborhood diagonal as fraction of core circumference [\%]",fontsize=setup["fontsize"])
        axes[1].scatter(ipt["simspaces"][0][:,4],sq_geodiff, s=1, c=cline[idx_sort_back], cmap=plt.cm.Spectral_r)
        axes[1].set_xlabel("Step in random path",fontsize=setup["fontsize"])
        axes[1].set_ylabel("Product of geographic coordinate differences [$deg^2$]",fontsize=setup["fontsize"])
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[0]), bbox_inches='tight', dpi = setup["dpi"])
            
    elif plot_type == 'kriging_weights':
        import numpy as np
        fig, axes = plt.subplots(1, 3, figsize=setup["figsize"])
        
        sections = setup["N"]/3
        subsections = sections/10
        
        for i in range(400,int(sections),int(subsections)):
            ic = 1-i/setup["N"]
            sp = i/setup["N"]*100
            axes[0].plot(np.array(ipt["kriging_weights"][i]).ravel(),label='%d%s' %(sp, '\%'), color = '%.3f' %ic)
            #axes[0].legend(loc='best',fontsize=setup["fontsize"])
            axes[0].set_xlabel('simulation data idx',fontsize=setup["fontsize"])
            axes[0].set_ylabel('Kriging weight',fontsize=setup["fontsize"])
        for i in range(int(sections+100),int(sections*2),int(subsections*2)):
            ic = 1-i/setup["N"]
            sp = i/setup["N"]*100
            axes[1].plot(np.array(ipt["kriging_weights"][i]).ravel(), label='%d%s' %(sp, '\%'), color = '%.3f' %ic)
            #axes[1].legend(loc='best',fontsize=setup["fontsize"])
            axes[1].set_xlabel('simulation data idx',fontsize=setup["fontsize"])
            #axes[1].set_ylabel('weight',fontsize=setup["fontsize"])
        for i in range(int(sections*2+100),setup["N"],int(subsections)):
            ic = 1-i/setup["N"]
            sp = i/setup["N"]*100
            axes[2].plot(np.array(ipt["kriging_weights"][i]).ravel(), label='%d%s' %(sp, '\%'), color = '%.3f' %ic)
            #axes[2].legend(loc='best',fontsize=setup["fontsize"])
            axes[2].set_xlabel('simulation data idx',fontsize=setup["fontsize"])
            #axes[2].set_ylabel('weight',fontsize=setup["fontsize"])
            
        fig.legend(loc='right',fontsize=setup["fontsize"]) #, bbox_to_anchor=(0.9, 0.5)
        plt.subplots_adjust(right=0.86)
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[0]), bbox_inches='tight', dpi = setup["dpi"])
            
    elif plot_type == 'kriging_weights_rel':      
        import numpy as np
        plt.figure(figsize=setup["figsize"])
        steps = setup["N"]/20
        for i in range(400,setup["N"],int(steps)):
            ic = 1-i/setup["N"]
            sp = i/setup["N"]*100
            plt.semilogy(abs(np.array(ipt["kriging_weights_rel"][i]).ravel()),label='%d%s' %(sp, '\%'), color = '%.3f' %ic)
            plt.xlabel('simulation data idx',fontsize=setup["fontsize"])
            plt.ylabel('$|$Kriging weight $\cdot$ simulation data$|$',fontsize=setup["fontsize"])
            
        plt.legend(loc='right',fontsize=setup["fontsize"], bbox_to_anchor=(1.2, 0.5)) #, 
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[0]), bbox_inches='tight', dpi = setup["dpi"])    
            
    elif plot_type == 'kriging_weights_short':
        import numpy as np
        fig, axes = plt.subplots(1, 3, figsize=setup["figsize"])
        labels = ["10\%", "25\%", "50\%", "75\%", "100\%"]
        clrs = [1-0.1, 1-0.25, 1-0.5, 1-0.75, 1-0.9]
        
        for i in range(0,2):
            axes[0].plot(np.array(ipt["kriging_weights"][i]).ravel(),label='%s' %labels[i], color = '%.3f' %clrs[i])
            axes[0].set_xlabel('simulation data idx',fontsize=setup["fontsize"])
            axes[0].set_ylabel('Kriging weight',fontsize=setup["fontsize"])
        for i in range(2,4):
            axes[1].plot(np.array(ipt["kriging_weights"][i]).ravel(),label='%s' %labels[i], color = '%.3f' %clrs[i])
            axes[1].set_xlabel('simulation data idx',fontsize=setup["fontsize"])
        
        i = 4
        axes[2].plot(np.array(ipt["kriging_weights"][i]).ravel(),label='%s' %labels[i], color = '%.3f' %clrs[i])
        axes[2].set_xlabel('simulation data idx',fontsize=setup["fontsize"])
            
        fig.legend(loc='right',fontsize=setup["fontsize"]) #, bbox_to_anchor=(0.9, 0.5)
        plt.subplots_adjust(right=0.86)
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[0]), bbox_inches='tight', dpi = setup["dpi"])  
            
    elif plot_type == 'kriging_weights_rel_short':      
        import numpy as np
        plt.figure(figsize=setup["figsize"])
        labels = ["10\%", "25\%", "50\%", "75\%", "100\%"]
        clrs = [1-0.1, 1-0.25, 1-0.5, 1-0.75, 1-0.9]
        for i in range(0,5):
            plt.semilogy(abs(np.array(ipt["kriging_weights_rel"][i]).ravel()),label='%s' %labels[i], color = '%.3f' %clrs[i])
            plt.xlabel('simulation data idx',fontsize=setup["fontsize"])
            plt.ylabel('$|$Kriging weight $\cdot$ simulation data$|$',fontsize=setup["fontsize"])
            
        plt.legend(loc='right',fontsize=setup["fontsize"], bbox_to_anchor=(1.2, 0.5)) #, 
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[0]), bbox_inches='tight', dpi = setup["dpi"])         
            
    elif plot_type == 'sim_vals':      
        import numpy as np
        plt.figure(figsize=setup["figsize"])
        steps = setup["N"]/20
        for i in range(400,setup["N"],int(steps)):
            ic = 1-i/setup["N"]
            sp = i/setup["N"]*100
            plt.semilogy(abs(np.array(ipt["Zi"][i]).ravel()),label='%d%s' %(sp, '\%'), color = '%.3f' %ic)
            plt.xlabel('simulation data idx',fontsize=setup["fontsize"])
            plt.ylabel('$|$Simulation data$|$',fontsize=setup["fontsize"])
            
        plt.legend(loc='right',fontsize=setup["fontsize"], bbox_to_anchor=(1.2, 0.5)) #, 
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[0]), bbox_inches='tight', dpi = setup["dpi"]) 
            
    elif plot_type == 'prior_fit_diag':
        import numpy as np
        from matplotlib.font_manager import FontProperties
        fig, axes = plt.subplots(len(ipt), 2, figsize=(16,16))
        font=FontProperties()
        font.set_size('x-large')
        
        # PRIOR HISTOGRAM FIT
        for i in range(0,len(ipt)):
            if ipt[i]["realization amount"] > 1:
                for j in range(0,ipt[i]["realization amount"]-1):
                    y,binEdges=np.histogram(ipt[i]["realizations"][:,j],bins=bins)
                    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                    axes[i,0].plot(bincenters,y,'-',color = '0.75')
            else:
                j = -1
                
            y,binEdges=np.histogram(ipt[i]["realizations"][:,j+1],bins=bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[i,0].plot(bincenters,y,'-',color = '0.75',label='Realizations')    
    
            y,binEdges=np.histogram(args[3],bins=bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[i,0].plot(bincenters,y,'k--',label='Prior w/o dipole')
            
            maxx = 0.25*np.max(bincenters)
            maxy = 0.75*np.max(y)
            
            if args[6] == "C":
                n_test = int(np.round(setup["N"]/ipt[i]["SN"]))
                stattext = '\\textbf{$%s$}\n$N_{nsv} = N_S/%d = %s$' %(args[5][i], n_test, ipt[i]["SN"])
                axes[i,0].text(maxx, maxy, stattext, fontproperties=font)
            elif args[6] == "S":
                n_test = int(np.round(args[7]["N"]/ipt[i]["DN"]))
                stattext = '\\textbf{$%s$}\n$N_{nsv} = N_D/%d = %s$' %(args[5][i], n_test, ipt[i]["DN"])
                axes[i,0].text(maxx, maxy, stattext, fontproperties=font)
            elif args[6] == "CS":
                sn_test = int(np.round(setup["N"]/ipt[i]["SN"]))
                dn_test = int(np.round(args[7]["N"]/ipt[i]["DN"]))
                stattext = '\\textbf{$%s$}\n$N_{nsv} = N_S/%d = %s$\n$N_{obs} = N_D/%d = %s$' %(args[5][i], sn_test, ipt[i]["SN"], dn_test, ipt[i]["DN"])
                axes[i,0].text(maxx, maxy, stattext, fontproperties=font)
                
            y,binEdges=np.histogram(args[0],bins=bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[i,0].plot(bincenters,y,'k-',label='Prior w/ dipole')
            

            
            #stddev_p = np.std(args[0])
            #var_p = np.var(args[0])
            #mean_p = np.mean(args[0])
            #stattext = '\\textbf{Prior}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
            #axes[0,1].text(2000000, 350, stattext, fontproperties=font)
            
            #stddev = np.std(ipt[i]["realizations"])
            #var = np.var(ipt[i]["realizations"])
            #mean = np.mean(ipt[i]["realizations"])
            
            #minx = np.min(bincenters)
            #maxy = 0.5*np.max(y)
            
            #stattext = '\\textbf{Prior}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Realization mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_p, stddev_p, var_p, mean, stddev, var)
            #axes[i,0].text(minx, maxy, stattext, fontproperties=font)
            
                
            # SEMI-VARIOGRAM FIT
            if ipt[i]["realization amount"]>1:
                for j in range(0,ipt[i]["realization amount"]-1):
                    axes[i,1].plot(ipt[i]["realizations sv lags"],ipt[i]["realizations sv"][:,j],color = '0.75')
            else:
                j = -1
            axes[i,1].plot(ipt[i]["realizations sv lags"],ipt[i]["realizations sv"][:,j+1],color = '0.75',label='Realizations')
            if prior_dipole == True:
                axes[i,1].plot(args[2]["total data lags"],args[2]["total data sv"],'.', color='C2',markersize=2,label='Semi-variogram w/ dipole')
            axes[i,1].plot(args[1]["total data lags"],args[1]["total data sv"],'.',markersize=2,label='Semi-variogram w/o dipole')
            axes[i,1].plot(ipt[i]["realizations sv lags"],ipt[i]["realizations mean sv"],'k-',label='Realization mean')
            axes[i,1].plot(args[1]["sv model x"],args[1]["sv model y"],color='C1', linewidth = 2, label='%s model' % args[1]["model names"][args[1]["sv model"]])
            
        axes[0,1].legend(loc='upper center',fontsize=setup["fontsize_sub"], markerscale=4, bbox_to_anchor=(0.45,1.85))  
        axes[0,0].legend(loc='upper center',fontsize=setup["fontsize_sub"], bbox_to_anchor=(0.45,1.63))   
        
        axes[len(ipt)-1,0].set_xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        fig.text(0.09, 0.5,'Bin count',fontsize=setup["fontsize"], ha='center', va='center', rotation='vertical')
        fig.text(0.52, 0.5,'Semi-variance $%s^2$' % setup["unit"],fontsize=setup["fontsize"], ha='center', va='center', rotation='vertical')    
        axes[len(ipt)-1,1].set_xlabel('Lag [km]',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[4]), bbox_inches='tight', dpi = setup["dpi"])

    elif plot_type == 'lsq_fit_diag':
        import numpy as np
        from matplotlib.font_manager import FontProperties
        if type(ipt) is dict:
            ipt_save = ipt
            ipt = list()
            ipt.append(ipt_save)
            ipt.append(ipt_save)
        len_ipt = len(ipt)
        fig, axes = plt.subplots(len_ipt, 2, figsize=(26,35))
        font=FontProperties()
        font.set_size('x-large')

        for i in range(0,len_ipt):
            # DATA HISTOGRAM FIT
            if ipt[i]["realization amount"] > 1:
                for j in range(0,ipt[i]["realization amount"]-1):
                    y,binEdges=np.histogram(ipt[i]["data prediction"][:,j],bins=200)
                    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                    axes[i,0].plot(bincenters,y,'-',color = '0.75')
            else:
                j = -1

            y,binEdges=np.histogram(ipt[i]["data prediction"][:,j+1],bins=200)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[i,0].plot(bincenters,y,'-',color = '0.75')    
            
            y,binEdges=np.histogram(args[3],bins=200)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[i,0].plot(bincenters,y,'k-',label='Target')
            
            maxy = 0.65*np.max(y)
            
            y,binEdges=np.histogram(np.mean(ipt[i]["data prediction"],axis=1),bins=200)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[i,0].plot(bincenters,y,'r--',label='LSQ prediction', linewidth=1)
            
            axes[i,0].legend(loc='upper left',fontsize=setup["fontsize_sub"])
            axes[i,0].set_xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
            axes[i,0].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
            
            stddev_d = np.std(args[3])
            var_d = np.var(args[3])
            mean_d = np.mean(args[3])
            #stattext = '\\textbf{Data}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
            #axes[0,0].text(-15000, 160, stattext, fontproperties=font)
            
            stddev = np.std(ipt[i]["data prediction"])
            var = np.var(ipt[i]["data prediction"])
            mean = np.mean(ipt[i]["data prediction"])
            
            minx = np.min(bincenters)
            
            
            #stattext = '\\textbf{Data prediction mean}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
            
            stattext = '\\textbf{Observations}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{Prediction}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_d, stddev_d, var_d, mean, stddev, var)
            axes[i,0].text(minx, maxy, stattext, fontproperties=font)


            # DATA RESIDUAL
            axes[i,1].set_xlabel('Data residuals, $B(\mathbf{r})_{obs} - B(\mathbf{r})_{est}$  %s' %setup["unit"], fontsize=setup["fontsize"],labelpad=setup["gensize"])
    
            if ipt[i]["realization amount"] > 1:
                for j in range(0,ipt[i]["realization amount"]-1):
                    y,binEdges=np.histogram(ipt[i]["residual_forward"][:,j],bins=bins)
                    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
                    axes[i,1].plot(bincenters,y,'-',color = '0.75')
            else:
                j = -1
            
            #maxy = np.max(y)
            
            y,binEdges=np.histogram(ipt[i]["residual_forward"][:,j+1],bins=bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[i,1].plot(bincenters,y,'-',color = '0.75')
            
            minx = np.min(ipt[i]["residual_forward"])
            maxy = 7.5*np.mean(y)
            
            y,binEdges=np.histogram(ipt[i]["residual_forward_mean"],bins=bins)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            axes[i,1].plot(bincenters,y,'k-',label='Residuals')
            axes[i,1].legend(loc='upper right',fontsize=setup["fontsize_sub"])
            axes[i,1].set_ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
            stddev_m = np.std(ipt[i]["residual_forward_mean"])
            var_m = np.var(ipt[i]["residual_forward_mean"])
            mean_m = np.mean(ipt[i]["residual_forward_mean"])   
            
            stattext = '\\textbf{Statistics}\n$\mu=%.3g$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_m, stddev_m, var_m)
            axes[i,1].text(minx, maxy, stattext, fontproperties=font)
        
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[4]), bbox_inches='tight', dpi = setup["dpi"])
            
    elif plot_type == "dip_no_dip_hist":
        import numpy as np
        from matplotlib.font_manager import FontProperties
        #fig, axes = plt.subplots(1, 1, figsize=(20,20))
        plt.figure(figsize = setup["figsize"])
        font=FontProperties()
        font.set_size('xx-large')
        
        y,binEdges=np.histogram(ipt["data"],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'k--',label='Prior w/o dipole')
        
        y,binEdges=np.histogram(args[0]["data"],bins=bins)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,'k-',label='Prior w/ dipole')
        plt.legend(loc='upper right',fontsize=setup["fontsize_sub"])
        plt.xlabel('Field value %s' % setup["unit"],fontsize=setup["fontsize"],labelpad=setup["gensize"])
        plt.ylabel('Bin count',fontsize=setup["fontsize"],labelpad=setup["gensize"])
        
        stddev_p = np.std(args[0]["data"])
        var_p = np.var(args[0]["data"])
        mean_p = np.mean(args[0]["data"])
        #stattext = '\\textbf{Prior}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean, stddev, var)
        #axes[0,1].text(2000000, 350, stattext, fontproperties=font)
        
        stddev = np.std(ipt["data"])
        var = np.var(ipt["data"])
        mean = np.mean(ipt["data"])
        
        minx = np.min(bincenters)
        maxy = 0.5*np.max(y)
        
        stattext = '\\textbf{w/ dipole}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$\n\\textbf{w/o dipole}\n$\mu=%.3e$\n$\sigma=%.3g$\n$\sigma^2=%.3g$' %(mean_p, stddev_p, var_p, mean, stddev, var)
        plt.text(minx, maxy, stattext, fontproperties=font)
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = setup["gensize"])
        
        if setup["savefig"] == True: 
            plt.savefig('%s%s_%s.png' % (save_path, plot_type, args[1]), bbox_inches='tight', dpi = setup["dpi"])
        plt.show()
            
    else:
        print('Please specify plot type as one of the following options:')
        print('')
        print(' -  target histogram')
        print(' -  grid')
        print(' -  data_basemap')
        print(' -  target_semi_variogram')
        print(' -  model_semi_variogram')
        print(' -  vario LUT')
        print(' -  data vs normal QFs')
        print(' -  data normal score transform')
        print(' -  normal QF range')
        print(' -  conditional QF range')
        print(' -  mean and var QF coverage')
        print(' -  realization_basemap')
        print(' -  realization_histogram')
        print(' -  prediction_histogram')
        print(' -  realization_semi_variogram')
        print(' -  realization_semi_variogram_pc')
        print(' -  data_misfit') 
        print(' -  prior_misfit') 
        print(' -  fit_diag')
        print(' -  neighborhood_size')
        print(' -  kriging_weights')
        print(' -  kriging_weights_rel')
    return