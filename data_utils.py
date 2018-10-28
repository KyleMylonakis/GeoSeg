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

def ground_truth_1d_2layer_single_example(labels_tensor,output_shape=100):
        """
        Assumes labels_tensor is a numpy array of the form (c1,c2,ratio) 
        corresponding to the ground truth where the top 100*ration % of 
        pixels are c1 and the bottom 100*(1-ratio)% are c2. Returns an 
        output_shape long tensor with a 1 for high speed and 0 for low speed. 

        Parameters:
        -----------
                labels_tensor: np.array of shape (3,)
        """

        wave_speeds = labels_tensor[:-1]
        ratio = labels_tensor[-1]

        assert  0.0 < ratio < 1.0, 'Ratio must be in (0,1) but got %f'%ratio

        wave_speeds = list(wave_speeds)

        split = int(output_shape*ratio)
        
        ground_truth = [0]*output_shape
        ground_truth[:split] = [wave_speeds[0]]*split
        ground_truth[split:] = [wave_speeds[1]]*(output_shape - split)
        
        wave_speeds.sort()      # modifies wave speeds
        ground_truth = [wave_speeds.index(x) for x in ground_truth]

        return ground_truth

tests = np.array([[.1,.3,.5],
                  [0.3,0.1,0.9],
                  [1.0,2.1,1.2] ])

for i in range(3):
        test = tests[i,:]
        print(ground_truth_1d_2layer_single_example(test,output_shape=11))

def ground_truth_1d_2layer(labels_tensor,output_shape=100):
        """
        Assumes labels_tensor is of the form (num_samples, c1, c2, ratio) corresponding
        to the ground truth where the top 100*ration % of pixels are c1 and 
        the bottom 100*(1-ratio)% are c2. Returns an (num_samples, output_shape) long tensor 
        with a 1 for high speed and 0 for low speed. 
        """

        ground_truth = np.zeros((output_shape,), dtype = np.float32)
        

        return None
