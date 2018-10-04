import numpy as np 

# Rewrite in a parallel or vectorized manner Map/Reduce formalism for example
# May have to unroll the loops or something, or think about cache performance here.
def interface_groundtruth_1d(y,
                            output_shape=100,
                            std_ratio=0.1):

    interface_depth = y[:,2]
    #total_depth = feature["total_depth"]
    total_depth = 20.0
    std = std_ratio*total_depth
    var = std**2
    #print('var: ', var)

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
    #total_depth = 9999.0
    total_depth = 200.0
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

