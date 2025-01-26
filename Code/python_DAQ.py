import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os

DONT_USE_SECOND_PEAK = False

script_directory = os.path.dirname(os.path.abspath(__file__))

files = [f for f in os.listdir(script_directory) if os.path.isfile(os.path.join(script_directory, f))]
box_n = []

i=0
while i < len(files): #find the data and motion files
    if files[i][:4] == "data":
        box_n.append(int(files[i].strip('_.datcsv')))
    if files[i][:6] == "motion":
        box_n.append(int(files[i].strip('_.motinb')))
    i+=1

box_n = list(set(box_n)) #list of unique box numbers
box_n.sort()

print("All boxes:",box_n)
        
names = box_n
for i in range(len(names)):
    names[i] = str(names[i])

trials = len(names)
lengths = []

all_comp = np.nan #all the computed lenghts from the frequecies
err_comp = np.nan #error on the computed lengths

def f(l, k, c, m=0.638, s=0.533): #function for frequency as a function of length
    '''
    l = measured length
    k = 3EI = spring "constant"
    c = c #note that the c is squared, so a negative value is acceptable (the same as a positive)
    m = mass of the electronics
    s = linear mass density of the pole
    '''
    
    return np.sqrt((k*((m+s*l)**2)/((m*l+s*(l**2)/2)**3)) - ((c/(m+s*l))**2) )

def zero_fffunc(l, freq_solve, k=182.64026108, c=3.91926779):
    """ function allowing to find the the length from the frequency"""
    return np.abs(freq_solve - f(l, k, c))

def norm_harmonic(max_freqs_data):
    """
    Function that takes the n x 2 array containing the frequency peaks
    Returns the rectified frequencies (brought back down to what is probably
    the lowest harmonic of the oscillation)
    """
    all_rec_freq_data = np.zeros(np.shape(max_freqs_data)[0])
    for m in range(np.shape(max_freqs_data)[0]):
        max_freqs = max_freqs_data[m]
        first_zero = False
        second_zero = False

        if max_freqs[0]==0: #looks to see if some frequencies == 0
            first_zero = True
        elif max_freqs[1]==0:
            second_zero = True
        else:
            ratio = max_freqs[0]/max_freqs[1]
            
        if first_zero and second_zero: #if both are 0, null result
            rec_freq = np.nan
        elif first_zero: #if one peak is 0, only the other is good data
            rec_freq = max_freqs[1]
        elif second_zero: #if one peak is 0, only the other is good data
            rec_freq = max_freqs[0]
        elif abs(ratio-0.5)<0.05: #looking for which peak is the bigger one
            rec_freq = (max_freqs[0] + (max_freqs[1]/2))/2
        elif abs(ratio-2)<0.15:
            rec_freq = ((max_freqs[0]/2) + max_freqs[1])/2
        else: #in case the peaks are simply not related (probably bad data)
            rec_freq = np.nan
                    
        all_rec_freq_data[m] = rec_freq
        
    all_rec_freq_data[all_rec_freq_data==0] = np.nan
    all_rec_freq_data = all_rec_freq_data[~np.isnan(all_rec_freq_data)]
    all_rec_freq_data = np.sort(all_rec_freq_data)
    
    return all_rec_freq_data

def sample_selection(all_rec_freq_data):
    """
    Takes in all the rectified data from a given sample
    Outputs new rectified frequencies based on what should be
    the lowest harmonic frequency of this much larger set of data
    """
    n = np.shape(all_rec_freq_data)[0]
    if n > 3: #if the set of data is large enough, tries to bring all the frequencies back down to lowest frequency
        mid = n//2 - 1
        mod = all_rec_freq_data[mid]
            
        for v in range(n):  
            ratio = all_rec_freq_data[v]/mod
                
            #this does the same data selection as before, but on the entire set as opposed to the 2 main peaks
            if abs(ratio-0.5)<0.05:
                all_rec_freq_data[v] = all_rec_freq_data[v]*2
            elif abs(ratio-2)<0.15:
                all_rec_freq_data[v] = all_rec_freq_data[v]/2
            elif not(abs(ratio-1)<0.3):
                #print("Deleted", all_rec_freq_data[v])
                all_rec_freq_data[v] = np.nan
                 
        all_rec_freq_data = all_rec_freq_data[~np.isnan(all_rec_freq_data)]
        
        comp_lens = []
        for w in range(np.shape(all_rec_freq_data)[0]): #uses the Length v. freq function to determine the length from each freq
            new_len = sp.fsolve(zero_fffunc, [0.5], args=(all_rec_freq_data[w]))[0]
            comp_lens.append(new_len)
                
        comp_lens = np.array(comp_lens) #all computed lengths
        all_comp = np.nanmean(comp_lens) #mean length for a given run
        err_comp = np.nanstd(comp_lens)/np.sqrt(np.shape(comp_lens)[0]) #error on the mean length
        
    else:
        #print("Inconclusive set")
        all_comp = np.nan  
        err_comp = np.nan
        
    return all_comp, err_comp


### main code ######################################################################

data_dtype = np.dtype([('epoch_time', np.int32),('vReal', np.float32, (2048,))])

for u in range(np.shape(box_n)[0]):
    
    data_tit = "data_" + names[u] + ".csv"
    motion_tit = "motion_" + names[u] + ".bin"

    all_data = np.genfromtxt(data_tit, delimiter = ",")
    all_motion = np.fromfile(motion_tit, dtype=data_dtype)
    all_motion = np.column_stack((all_motion['epoch_time'], all_motion['vReal']))
    
    data_times = all_data[:,2]
    motion_times = all_motion[:,0]
    times = np.append(data_times, motion_times)
    times = np.sort(np.array(list(set(times))))
    
    final_data = np.zeros((len(times), 8)) #for csv file: time, temperature, motion length, error, data length, error, depth sensor length, error
    final_data[:,0] = times
    
    all_motion = all_motion[:,1:-1] # shape (n, 2048) skips nan values at the end and time values in the beginning
    
    for time in times:
    
        motion = all_motion[motion_times==time,:]
        data = all_data[data_times==time,:]
        
        max_freqs_data = data[:,4:6]*2 #get the two main frequencies, they need to be multiplied by two to work with other standards in the code
        if DONT_USE_SECOND_PEAK: #can be useful if the second peak is bad / wrong
            max_freqs_data[:,1]=0
        
        all_rec_freq_data = norm_harmonic(max_freqs_data)
        
        all_comp, err_comp = sample_selection(all_rec_freq_data)
        
        final_data[final_data[:,0]==time,4] = all_comp
        final_data[final_data[:,0]==time,5] = err_comp
        
        MOTION_LEN = np.shape(motion)[1]
        
        window = np.blackman(MOTION_LEN)
        motion *= window #makes the data cleaner

        fft_result = np.fft.fft(motion, axis=1)
        freq_x = np.fft.fftfreq(MOTION_LEN, 1/200) #frequencies on the x axis

        amp_fft = np.abs(fft_result)[:,:len(freq_x)//2] #amplitudes found via the fft
        pos_freq = freq_x[:len(freq_x)//2] #ordered frequencies on the x axis

        index_list = np.arange(0,MOTION_LEN//2)

        all_freqs_motion = np.zeros((np.shape(motion)[0],2))

        for i in range(0, np.shape(motion)[0]): #iterates over all data samples

            max_amp = 0
            max_i = np.array([0,0])
            current_max_i = 0
            for l in range(2): #finds the 2 main peaks for every data sample
                avg = 0
                for k in range(2, MOTION_LEN//5): #DATA_LEN//5 means that we ignore very high frequencies
                    avg += amp_fft[i,k] #finds the average
                avg /= (MOTION_LEN//5) - 2
                for k in range(2, MOTION_LEN//5):
                    near1 = abs(k-max_i[0]) <= 3
                    near2 = abs(k-max_i[1]) <= 3
                    #near3 = abs(k-max_i[2]) <= 3
                    if not (near1 or near2):
                        if amp_fft[i,k] > max_amp and amp_fft[i,k] >= 10*avg and amp_fft[i,k] > amp_fft[i,k+1] and amp_fft[i,k] > amp_fft[i,k-1]:
                            # if it has max amplitude, is not noise, and is actully a local max
                            max_amp = amp_fft[i,k]
                            current_max_i = k
                max_i[l] = current_max_i
                current_max_i = 0
                max_amp = 0

            max_freqs = max_i*200/(MOTION_LEN//2)
            if DONT_USE_SECOND_PEAK:
                max_freqs[1] = 0
            
            all_freqs_motion[i] = max_freqs
        
        all_freqs = norm_harmonic(all_freqs_motion)
        
        all_comp, err_comp = sample_selection(all_freqs)
        
        #for csv file: time, temperature, motion length, error, data length, error, depth sensor length, error
        final_data[final_data[:,0]==time,1] = np.mean(data[:,3])
        
        final_data[final_data[:,0]==time,2] = all_comp
        final_data[final_data[:,0]==time,3] = err_comp
        
        final_data[final_data[:,0]==time,6] = np.nanmean(data[:,6])/100
        final_data[final_data[:,0]==time,7] = np.nanstd(data[:,6])/np.sqrt(np.shape(data[:,6]))/100

    final_file_title = 'final_data_'+ str(box_n[u]) +'.csv'
    np.savetxt(final_file_title, final_data, delimiter=",")
    print(final_file_title, "saved, shape", np.shape(final_data))
    print("end")























