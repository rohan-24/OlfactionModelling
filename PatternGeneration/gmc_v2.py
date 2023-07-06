''' implement a greedy max cut sorting of distances into Hc/Mc patterns.
    Robert Lindroos 2020-12-08:->
'''

from time import time
import numpy as np

import pickle, json, sys
import pandas as pd
import create_distance_matrix as dmat

from numba import jit

import gmc_plotting as plot
import matplotlib.pyplot as plt



features = ['Pleasantness', 'Intensity', 'Familiarity', 'Edability']
PATH = '/home/rohan/Documents/Olfaction/'

np.random.seed(0) 
####Sigmoid Type 1
E = 3   #For sigmoid transformation
m = 0.4 #For sigmoid transformation


#####Sigmoid Type 2 (Gompertz Function)
beta = 5  #Controls initial lag
gamma = 10  #Controls speed of growth


####Factor by which to reduce overall overlap, see greedy update
dampen_factor = 2.75

### Flag for saving plot in each iteration for animation
save4animation = 0
animation_plot_counter = 0

def run_all_individual(emax=0.5, min_index=1, max_index=37):
    # this is runs distance matrices of all individuals in a range
    #   (indexed 1-37 in the original data)
    
    # distances sorted on perceptual features (features[i])
    for i in range(2):
        for fp in range(min_index,max_index+1):
            main(non_standard=1, i=i, fp=fp, emax=emax)



def get_W2V_dmat():
    #w2v_simmat = pd.read_csv(PATH+'W2V/BloggModel_NewTotalWords_distancesCleaned.csv')
    #w2v_simmat = pd.read_csv(PATH+'W2V/BloggModel_CriticalWordsEng_distances.csv')

    ############# 16 Odors
    # w2v_simmat = pd.read_csv(PATH+'W2V/BloggModel_CriticalWords_similarity_sorted(SNACK).csv',index_col=0)
    # distmat = 1-w2v_simmat

    ############# top 3 descriptors 35 odors
    # distmat = pd.read_csv(PATH+'W2V/top3snackdescriptors_distmat_eng.csv',index_col=0)

    ############# threshold based top descs (see SnackAnalysis.extract_top_descs())
    #distmat = pd.read_csv(PATH+'W2V/topdescs0.5thresh_distmat.csv',index_col=0)
    #distmat = pd.read_csv(PATH+'W2V/topdescs0.33thresh_distmat.csv',index_col=0)
    distmat = pd.read_csv(PATH+'W2V/si_nclusters4_topdescs_distmat.csv',index_col=0)
    # distmat = pd.read_csv(PATH+'W2V/si_nclusters4_topdescs_distmat_Age&MMSEFilteredSNACKData.csv',index_col=0)
    
    #distmat = pd.read_csv(PATH+'W2V/correctdescs_maxassocs4_distmat.csv',index_col=0)  

    ############# All words
    # distmat = pd.read_csv(PATH+'W2V/Top90PercentileDescs_Distances.csv',index_col=0)  
    
    # w2v_simmat *= 10 #Scale similarity

    #distmat[((distmat>0.5) & (distmat<0.9))] = 2
    # distmat =np.log(distmat+1)

    # distmat = (distmat - distmat.values.min())/(distmat.values.max()-distmat.values.min())  
    # distmat = (np.sin(6*distmat)+1)/2  
    # np.fill_diagonal(distmat.values, 0)      

   
    #distmat = (distmat - distmat.values.min())/(distmat.values.max()-distmat.values.min()) 
    #distmat = 1./(1 + (m/distmat)**E)   ######Sigmoid Type 1
    #distmat = np.exp(-1*np.exp(beta-gamma*distmat)) #####Sigmoid Type 2
    D_final = np.array(distmat)*10

    #D_final = 5 * D_scaled / np.max(D_scaled)    
    #print(distmat)
    #distmat = pd.read_csv(PATH+'W2V/semanticneighbours_n3.csv',index_col=0)
    #distmat = pd.read_csv(PATH+'W2V/top3snackdescriptors_distmat.csv',index_col=0)
    #dissim_mat = np.arccos(w2v_simmat.iloc[:,2:].to_numpy())/np.pi
    #dissim_mat = (1-w2v_simmat.iloc[:,2:]).to_numpy()
    #return dissim_mat,w2v_simmat.Words.to_numpy()       #Return matrix and labels
    return D_final#distmat.index.to_numpy()

def get_odor_dmat():

    simmat = pd.read_csv('/home/rohan/Documents/Olfaction/Odor_similarity_data/Odors_MeanRatedSimilarity.csv',index_col=0)
    distmat = 1-simmat

    #distmat = np.log(distmat+1)
    #distmat = 1./(1 + (m/distmat)**E)   ###Sigmoid type 1
    # distmat = np.exp(-1*np.exp(beta-gamma*distmat)) ####Sigmoid type 2
    # D_scaled = np.array(distmat)*10
    # D_final = 10 * D_scaled / np.max(D_scaled) 
    D_final = np.array(distmat)*10
    return D_final#distmat.index.to_numpy()

def main(H=15, M=15, npat=5, mean_overlap=0, non_standard=0, i=None, fp=None, emax=20):    
    # based on a distance matrix (D):
    # 1. initialize. set p0 = 0 for all Hc. 
    #        Random values for other patterns gives best result, all zero also work.
    # 2. fill pattern matrix column-vice based on cost and overlap
    # 3. optimze: repeatedly flip trough the matrix (using greedy strategy)
    
    # create pattern
    #Dn = D_from_xy_plane()
    #check_trinequality(D)
    # pre1 and pre2
    #Dn = resort_distances(D)
    #check_trinequality(Dn)
    
    if non_standard==1:
        perc = features[i]
        path = '../../Box/Odor_similarity_data/Individual_distance_matrices/'
        path = 'InputDistances/'
        if not fp: fp = 6
        fname = '{}D_fp{}_{}.json'.format(path,fp,perc)
        save_fname = 'Patterns/All_participants/pattern_fp{}_{}'.format(fp,perc)
        with open(fname, 'r') as f:
            Dn = np.array(json.load(f))
    elif non_standard==2:
        Dn = dmat.matobj(npat, H, mean=mean_overlap).matrix  
    elif non_standard==3:
        obj = dmat.matobj(npat, H, mean=mean_overlap, initiate=False)
        obj.random_from_XYZ_plain()
        Dn = obj.matrix
    elif non_standard==4:
        obj = dmat.matobj(npat, H, mean=mean_overlap, initiate=False)
        points = [[i,i] for i in range(0,20,2)]
        obj.from_point_cloud(points, 3, H=H)
        Dn = obj.matrix   
    elif non_standard==5:
        obj = dmat.matobj(npat, H, initiate=False)
        ng = 3
        obj.multimodal( ngroups=ng,
                        max_between_percentage=10,
                        max_within_percentage=60,
                        min_within_percentage=40)
        Dn = obj.matrix  
        save_fname = 'OutputPatterns/multi_D{}_npat{}_ngroups{}_emax{}'.format(
                        non_standard, 
                        npat,
                        ng, 
                        emax).replace('.','')
    else:
            ##### CHANGE THIS DEPENDING ON IF YOU WANT TO GENERATE PATTERNS FOR THE ODOR NETWORK OR LANG NETWORK
            Dn = get_odor_dmat() #get_odor_dmat() #get_W2V_dmat()
            # #Dn = Dn*H
            # npat = len(Dn)


            npat = len(Dn)

    
    #check_trinequality(Dn)
    if non_standard not in [1,5]: 
        # save_fname = 'GMCPatterns/w2v_single_labels_npat{}_emax{}_H{}_M{}_dampen_factor{}'.format( 
        #                                                                 npat, emax, H,M,dampen_factor).replace('.','_') 

        save_fname = 'GMCPatterns/odors_npat{}_emax{}_H{}_M{}_dampen_factor{}'.format( 
                                                                        npat, emax, H,M,dampen_factor).replace('.','_') 


        # save_fname = 'GMCPatterns/w2v_si_4clusters_npat{}_emax{}_H{}_M{}_dampen_factor{}'.format( 
        #                                                                 npat, emax, H,M,dampen_factor).replace('.','_')   
        
        if save4animation:
            animation_fname = '/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/Figures/DualNet/NeuralCodingWorkshop2023/GMC_Animation/'+'w2v_single_labels_H{}_M{}_dampen_factor{}'.format(H,M,dampen_factor).replace('.','_')

        else:
            animation_fname = ''
        # save_fname = 'GMCPatterns/w2v_si_2clusters_npat{}_emax{}_H{}_M{}_dampen_factor{}'.format( 
        #                                                                 npat, emax, H,M,dampen_factor).replace('.','_')   
        #### v2 incorporated D_scaling factor, v3 incorporates reduction of absolute value of overlap in pattern by a factor of half



    #check_trinequality(Dn)
        
    # 1,2

    #print(Dn)
    P = set_patterns(Dn, H, M, Nr=5,animation_fname=animation_fname)

    #print(Dn)
    #with open('Pinit.pkl', 'wb') as outfile:
    #    pickle.dump({'D':D, 'Dn':Dn, 'P':P}, outfile)
    
    # 3
    # optimize
    upd_pattern(P,Dn, H, M, saveP=1, save_fname=save_fname, animation_fname=animation_fname,emax=emax)
    
    #plt.show()
    if non_standard==1:
        plt.close('all')
    else:
        pass
        #plt.show()



  

  
def upd_pattern(P,Dn, H, M, N=False, saveP=False, save_fname=False, animation_fname='', emax=0.5):    
    ''' 
    iteratively "flip" values of idividual units to minimize the global error
    
    easily gets stuck in local minima.
        random flipping is added to get out off such states
    
    emax sets the maximal mean error allowed. 
        A small value make it harder for the algorithm to compleate (it might even be impossible)
    '''
    global animation_plot_counter

    if not N:
        Nr = [9,9,7,7,5,5,4,3,2,2,1,1,1]  # [3,2,1,0,0,0] # 
        N  = len(Nr)
        randomize = True
    else:
        randomize = False
        
    error       = np.zeros(N+1)
    error[0]    = hamming_distance(P,Dn,H)

    if animation_fname:
        animation_plot_counter +=1
        print(f'Saving animation plot {animation_plot_counter}')
        plot.plot_(Dn,P, H, error=error, saveFig=1, save_fname=animation_fname+f'-{animation_plot_counter}',showfigtitle=f'Iteration {animation_plot_counter-1}')

    best        = 100      #initialize to high value (arbitrary)
    
    for i,n in enumerate(Nr):
        
        # update patterns
        if randomize:
            # set_patterns allow shuffeling...
            P       = set_patterns(Dn, H, M, P=P, Nr=n)
        else:
            #... while the greedy_update only, does not
            P       = greedy_update(Dn, P, H, M)
        
        error[i+1]  = hamming_distance(P,Dn,H)
        
        if animation_fname:
            animation_plot_counter +=1
            print(f'Saving animation plot {animation_plot_counter}')
            plot.plot_(Dn,P, H, error=error, saveFig=1, save_fname=animation_fname+f'-{animation_plot_counter}',showfigtitle=f'Iteration {animation_plot_counter-1}')

        print(i, error[i+1])
        
        # if stuck in local minima
        if error[i+1] == error[i]: 
            # add random perturbations. pick n x,y pairs randomly... 
            x = np.random.randint(P.shape[0], size=n)
            y = np.random.randint(P.shape[1], size=n)
            
            for j in range(len(x)):
                P[x[j],y[j]] = np.random.randint(M) # ...and flip to random position in Hc (minicolumn)
            
        
        elif error[i+1] < best:
            Pbest = P.copy()    # TODO: remove? the plotting uses the last P as for now...
            best = error[i+1]
        
        # randomly shuffle columns
        np.random.shuffle(P.T)
        
    # Added while statement so that the updating without shuffeling goes on until 
    #   the error is fixed (not changing)    
    e = error[-1]
    eprev = 100
    error = list(error)
    while not eprev == e: 
        P = greedy_update(Dn, P, H, M)
        eprev = e
        e = hamming_distance(P,Dn,H)
        
        if animation_fname:
            animation_plot_counter +=1
            print(f'Saving animation plot {animation_plot_counter}')
            plot.plot_(Dn,P, H, error=error, saveFig=1, save_fname=animation_fname+f'-{animation_plot_counter}',showfigtitle=f'Iteration {animation_plot_counter-1}')

        error.append(e)
        
        print(eprev, e)


    
    if saveP:
        
        if e < emax:
            # save patterns...
            # OBS! Pbest have been exchanged for P here!
            if save_fname:
                with open('{}.json'.format(save_fname), 'w') as f:
                    json.dump(P.astype(int).tolist(), f)
            else:
                with open('OutputPatterns/patterns_average.json', 'w') as f:
                    json.dump(P.astype(int).tolist(), f) 
            # and plot
            plot.plot_(Dn,P, H, error=error, saveFig=1, save_fname=save_fname)
            
        else:
            print('--- error={:.2f} > {} --- reinitiating -----'.format(e, emax))
            P = set_patterns(Dn, H, M, Nr=5)
            upd_pattern(P,Dn,H,M,saveP=1,save_fname=save_fname, emax=emax)
            


@jit(nopython=True)
def greedy_update(D, P, H, M):
    # loop and fill the pattern matrix column-vice (one Hc at a time)
    #   **** This is the core of the algorithm ****
    
    npat = len(D)
    # loop over patterns one Hc at the time
    for j in range(H):
        for i in range(npat):
            
            # calc optimal position of hypercolumn j in pattern i 
            #       (as the other patterns look right now)
            C = np.zeros(M)
            for m in range(M):
                P[i,j] = m
                
                # create matrix with hamming distances of all patterns in P
                for ii in range(npat):
                    for jj in range(ii):
                        o = 0
                        for h in range(H):
                            if P[ii,h] == P[jj,h]: o += 1   #count number of overlaps between patterns ii and jj
                        

                        # calc distance to D:
                        #   1. number of Hc's minus the overlap -> hamming distance. 
                        #   2. hamming distance minus target distance (from D) in relative terms ([H-o]/H)
                        h   = (H/dampen_factor-o)*10*dampen_factor/H - D[ii,jj]    ### H/dampen_factor to reduce absolute value of overlap while preserving relationship
                        #print('h:{}, H:{}, o:{}, D:{}'.format(h,H,o,D[ii,jj]))
                        if h < 0: h = h*(-1)    # absolute value of distance
                        C[m]  += h
                
            # select the value with the shortest distance
            P[i,j] = np.argsort(C)[0]
    
    return P


def set_patterns(D, H, M, P=[], Nr=0,animation_fname=''):
    # fill pattern matrix column-vice (one Hc at a time)
    #   **** This is the core of the algorithm ****
    # -> adds a portion af randomly set columns in each hypercolumn
    
    npat = len(D)
    global animation_plot_counter
    # initialize
    if not len(P):
        P       = np.random.uniform(M, size=(npat,H))
        P[0]    = np.zeros(H)
    
    P = greedy_update(D, P, H, M)

    if animation_fname:
        animation_plot_counter +=1
        print(f'Saving animation plot {animation_plot_counter}')
        plot.plot_(D,P, H, saveFig=1, save_fname=animation_fname+f'-{animation_plot_counter}',showfigtitle=f'Iteration {animation_plot_counter-1}')

    # randomly perturb patterns?
    if Nr:
        
        # create random matrix with same shape
        R = np.random.randint(M, size=P.shape)
        
        # create matrix with zeros and ones used to select subportions of P and R
        r01 = np.ones(P.shape)
        r01[:Nr,:] = 0
        r01 = np.random.permutation(r01.flatten()).reshape(P.shape)
        
        P = P*r01 + R*(1-r01)
    
    return P


def hamming_distance(a,D,H, return_distances=0):
    
    '''
    calculates the hamming distance of the patterns in "a" 
        and returns the average difference with "D". 
        the distance is normlalized to 10 Hc (since the input matrices are set in the range of 0-10)
    
    The hamming distance (d) is the number of units that are not the same in two patterns.
        -e.g. [1,2,3] and [2,2,3] would have a distance of 1 
            (H minus the number of overlap 3-2=1)
        
    Arguments:
    -D (npat x npat) = the original distance matrix that "a" is calculated from
 
    -a (npat x H)    = the array holding the unit indices for each pattern over hypercolumns.
         
    '''

    d = (a[:, None, :] != a).sum(2) *10 / a.shape[1]
    print('d: ',np.mean(d),'\n','D:',np.mean(D))


    if return_distances:
        return d, np.mean(np.abs(d-D))
    else:       
        return np.mean(np.abs(d-D))
          


def D_from_xy_plane():
    
    # create distance matrix satisfying triangle inequality
    
    X = np.random.randint(8, size=16)
    Y = np.random.randint(8, size=16)
    
    d = np.zeros((16,16))
    
    for i in range(1,16):
        for j in range(i):
            d[i,j] = np.sqrt( np.square(X[j]-X[i]) + np.square(Y[j]-Y[i]) )
            d[j,i] = d[i,j]
    
    # this could potentially destroy the inequality? Rounding does destroy
    d = np.ceil(d)  
    
    return d       



def resort_distances(D):
    
    # find pattern largest average overlap, p0
    mean = np.mean(D, axis=0)   # axis could be 1 as well, since symetric...
    p0 = np.argmin(mean)     
    
    # sort patter based on similarity with p0
    si = np.argsort(D[p0])
    
    # re-map D based on p0-pN (N=15)
    Dn = np.zeros(D.shape)
    for i in range(len(D)):
        for j in range(len(D)):
            Dn[si[i],si[j]] = D[i,j]
    
    return(Dn)
            


def check_trinequality(D):
    
    N = len(D)
    for i in range(N):
        for j in range(i):
            assert all(D[i,j] <= D[i,:] + D[:,j])
    


# if run from terminal...   ===============================================================
if __name__ == "__main__":
    
    if len(sys.argv) > 1: 
        if sys.argv[1].isnumeric():
            print('inne', sys.argv[1])
            main(emax=float(sys.argv[1]))
        else:
            print('ERROR: accept single numeric argument only.\nexample run:\n\tpython gmc_sorting.py\nor\n\tpython gmc_sorting.py 0.4')
            print('\ndefault maximal error (emax) = 0.5')
    else:
        main()
    