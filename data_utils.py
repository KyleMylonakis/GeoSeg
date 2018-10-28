import numpy as np 
import multiprocessing as mp 

# TODO: Rewrite in a parallel or vectorized manner Map/Reduce formalism for example
#       May have to unroll the loops or something, or think about cache performance here.
def interface_groundtruth_1d(y,
                            output_shape=100,
                            std_ratio=0.1):

    interface_depth = y[:,2]
    total_depth = 20.0
    std = std_ratio*total_depth
    var = std**2

    ground_truth = np.linspace(0.0,total_depth,num=output_shape)
    #ground_truth = np.reshape(ground_truth,[output_shape,1])

    out = np.zeros((y.shape[0], output_shape), dtype=np.float32)
    for jj in range(y.shape[0]):
        for kk in range(output_shape):
            out[jj,kk] = np.exp(-0.5*(ground_truth[kk]-interface_depth[jj])**2/var)
    
    return out

def interface_groundtruth_max(y,
                            output_shape=100,
                            std_ratio=0.1):

    interface_depth = y[:,2]
    print('==================INTERFACE DEPTH==========: ', interface_depth)
    # Total depth = 200 for P-wave, 9999.0 for PS wave
    #total_depth = feature["total_depth"]
    total_depth = 9999.0
    #total_depth = 200.0
    std = std_ratio*total_depth
    var = std**2
    #print('var: ', var)

    ground_truth = np.linspace(0.0,total_depth,num=output_shape)
    #ground_truth = np.reshape(ground_truth,[output_shape,1])

    out = np.zeros((y.shape[0], output_shape,2), dtype=np.float32)
    for jj in range(y.shape[0]):
        for kk in range(output_shape):
                if ground_truth[kk] > interface_depth[jj]:
                        out[jj,kk,1] = 1.0
                else:
                        out[jj,kk,0] = 1.0
    return out

def one_hot_it(idx,num_classes):
        result = [0]*num_classes
        result[idx] = 1
        return np.array(result)

def ground_truth_1d_2layer_single_example(labels_tensor,output_shape=100):
        """
        Assumes labels_tensor is a numpy array of the form (c1,c2,ratio) 
        corresponding to the ground truth where the top 100*ration % of 
        pixels are c1 and the bottom 100*(1-ratio)% are c2. Returns an 
        output_shape long tensor with a 1 for high speed and 0 for low speed. 

        Parameters:
        -----------
                labels_tensor: np.array of shape (3,)
        Ret
        """

        wave_speeds = labels_tensor[:-1]
        ratio = labels_tensor[-1]
        assert  0.0 < ratio < 1.0, 'Ratio must be in (0,1) but got %f'%ratio

        wave_speeds = list(wave_speeds)

        split = int(output_shape*ratio)
        

        wave_speeds.sort()      # modifies wave speeds

        ground_truth = [0]*output_shape
        ground_truth[:split] = [one_hot_it(wave_speeds.index(wave_speeds[0]),2)]*split
        ground_truth[split:] = [one_hot_it(wave_speeds.index(wave_speeds[1]),2)]*(output_shape - split)
                
        return ground_truth

def ground_truth_1d_2layer_single_example_packed(labels_tensor_output_shape):
        labels_tensor,output_shape = labels_tensor_output_shape
        return ground_truth_1d_2layer_single_example(labels_tensor,output_shape=output_shape)

def ground_truth_1d_2layer(raw_labels,output_shape=100, num_cores = 4):
        """
        Assumes raw_labels is of the form [(c1, c2, ratio)] corresponding
        to the ground truth where the top 100*ration % of pixels are c1 and 
        the bottom 100*(1-ratio)% are c2. Returns an (num_samples, output_shape) long tensor 
        with a 1 for high speed and 0 for low speed. 

        Parameters:
        -----------
                raw_labels: np.array of shape (num_samples,3). Each row should correspond to ground truth 
                        labels (c1,c2,ratio) as described above.
                output_shape: Desired shape of 1D output for ground truth.
                num_cores: (Deprecated) Number of cores to use in parallel processing.
        Returns:
        --------
                ground_truth tensor of size output_shape. Each row is a 1 or 0 depending on whether it 
                came from the higher or lower wavespeed region respectively. 
        """
        num_samples = raw_labels.shape[0]        
        
        # TODO: Look into why parallezation didn't improve performance.
        #data = zip(raw_labels,[output_shape]*num_samples)
        #with mp.Pool(processes = num_cores) as pool:
        #        ground_truth = pool.map(ground_truth_1d_2layer_single_example_packed,data)

        #pool.close()
        #pool.join()
        result = [0]*num_samples
        for i in range(num_samples):
                example = raw_labels[i]
                result[i] = ground_truth_1d_2layer_single_example(example,output_shape=output_shape)
        
        result = np.array(result)
        return result


# Tests ground_truth_1d_2layer and times it. 
# TODO: Remove when ready to commit to master. 
#       Maybe put in a test script.
"""
num_samples = 100000
np.random.seed(1)
test = np.random.rand(num_samples,3)



import time 


t0 = time.time()
z = ground_truth_1d_2layer(test,output_shape= 11)
party = time.time() - t0 

out_test = [0]*num_samples
t0 = time.time()
for i in range(num_samples):
        out = test[i,...]
        out_test[i] = ground_truth_1d_2layer_single_example(out,output_shape=11)
out_test = np.array(out_test)

normy = time.time() - t0

msg = " \n z shape: {z.shape} \n out shape: {out_test.shape} \n parralell time: {party} \n normal time: {normy}"

print(msg.format(z=z,out_test=out_test,party=party,normy=normy))

print(np.linalg.norm(out_test-z))
"""     

