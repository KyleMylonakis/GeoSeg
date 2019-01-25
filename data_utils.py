import numpy as np 
import multiprocessing as mp 

# TODO @kmylonakis: Rewrite in a parallel or vectorized manner Map/Reduce formalism for example
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
        classes = list(wave_speeds)     # copy to sort
        split = int(output_shape*ratio)
        
        classes.sort()

        ground_truth = [0]*output_shape
        ground_truth[:split] = [one_hot_it(classes.index(wave_speeds[0]),2)]*split
        ground_truth[split:] = [one_hot_it(classes.index(wave_speeds[1]),2)]*(output_shape - split)
                
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

def ground_truth_1d_multilayer_single_example(labels_tensor,output_shape=100, low_speed_pocket = False):
        """
        Assumes labels_tensor is a numpy array of the form 
                (c1,c2,...,cN,r1,r2,...,r(n-1)) 
        Corresponding to the wavespeeds in various regions seperated
        at the ri. Sorts the wavespeeds and classes are assigned in 
        descending order starting at 0. 

        Parameters:
        -----------
                labels_tensor: np.array of shape (2num_classes - 1,)
                output_shape: Shape of output array (int)
                low_speed_pocket: Whether to interpret data as low_speed_pocket or not. If true 
                        the output will have only 2 classes. 0 for the low speed and 1 for all other
                        regions.
        Returns:
        --------
                If low_speed_pocket is False an array of size (output_shape,num_classes), if it is True
                then size is (output_shape, 2).
        """

        assert labels_tensor.shape[0] % 2 == 1, 'Labels must be an odd number but got {}'.format(labels_tensor.shape[0])

        num_classes = int((labels_tensor.shape[0] + 1)/2)
        wave_speeds = list(labels_tensor[:num_classes])
        classes = list(wave_speeds)
        ratios = labels_tensor[num_classes:]

        for r in ratios:
                assert 0 < r < 1, 'Ratio must be in (0,1) but got %f'%r

        splits = [int(r*output_shape) for r in ratios]
        
        splits.sort()           # modifies splits
        classes.sort()      # modifies wave speeds
        
        # Make a labels map {ci: label}
        if low_speed_pocket:
                num_classes = 2
                low_speed = min(wave_speeds)
                labels_map = {w:0 for w in wave_speeds if w != low_speed}
                labels_map[low_speed] = 1
        else:
                labels_map = {w:classes.index(w) for w in wave_speeds}

        ground_truth = [0]*output_shape
        ground_truth[:splits[0]] = [one_hot_it(labels_map[wave_speeds[0]],num_classes)]*splits[0]
        for i in range(1,len(wave_speeds)-1):
                s1,s2 = splits[i-1], splits[i]
                ground_truth[s1:s2] = [one_hot_it(labels_map[wave_speeds[i]],num_classes)]*(s2-s1)
        ground_truth[splits[-1]:] = [one_hot_it(labels_map[wave_speeds[-1]],num_classes)]*(output_shape - splits[-1])
        
        return np.array(ground_truth)

# Not used
def ground_truth_1d_multi_layer_single_example_packed(labels_tensor_output_shape):
        labels_tensor,output_shape = labels_tensor_output_shape
        return ground_truth_1d_2layer_single_example(labels_tensor,output_shape=output_shape)

def ground_truth_1d_multilayer(raw_labels,output_shape=100, num_cores = 4, low_speed_pocket = False):
        """
        Essentially a wrapper to vectorize ground_truth_1d_multilayer_single_example.

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
                low_speed_pocket: Whether to interpret data as low_speed_pocket or not. If true 
                        the output will have only 2 classes. 0 for the low speed and 1 for all other
                        regions.
        Returns:
        --------
                If low_speed_pocket is False an array of size (num_samples,output_shape,num_classes), if it is True
                then size is (num_samples,output_shape, 2).
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
                result[i] = ground_truth_1d_multilayer_single_example(example,output_shape=output_shape, low_speed_pocket=low_speed_pocket)
        
        result = np.array(result)
        return result

# Wrap up the 1d pocket label function
def ground_truth_1d_pocket(raw_labels, output_shape=100):
        return ground_truth_1d_multilayer(raw_labels,output_shape=output_shape, low_speed_pocket= True)


def ground_truth_2d_circle_pocket_single_example(radius,x,z,
                                                 xmax = 2.0,
                                                 zmax = 3.0,
                                                 output_shape = [128,128], 
                                                 boundary_class = False):
        """
        Class map {0: low speed (pocket), 1: high speed, 2: boundary}
        Assumes (0,0) is in the upper left corner of a xmax by zmax rectangle.
        Creates a one hot vector at each pixel for a given class.

        Parameters:
        -----------
                radius: The radius of the pocket. (float)
                x: The x coordinate of the pocket. (float)
                z: The z coordinate of the pocket. (float)
                xmax: The length of the x direction. (float)
                zmax: The length of the z direction. (float)
                output_shape: The shape for generated output. Note that
                        [M,N] will have the M pixels in the z direction
                        and N pixels in the x direction. 
        Returns:
        --------
                A numpy array of shape [M,N,3] where p(i,j,k) = 1 means that 
                location (i,j) belongs to class k (i.e. it is a one hot vector).
        """
        _msg = "!!!You are using default parameters!!!! \nxmax:{} \nzmax:{}"
        if xmax == 2.0 and zmax == 3.0:
                print(_msg.format(2.0,3.0))
        # Get pixel value for x and z.
        # Note the order is switched!!!
        x_node, z_node = output_shape[1] * x/xmax, output_shape[0] * z/zmax
        x_step, z_step = xmax/float(output_shape[1]), zmax/float(output_shape[0]) # km/pixel in each direction

        # Find the boundary box
        x1, x2 = int(x_node - radius/x_step),int( x_node + radius/x_step)
        z1, z2 = int(z_node - radius/z_step), int(z_node+radius/z_step)
        
        xbox = [max(x1,0), min(x2,output_shape[1])]
        zbox = [max(z1,0), min(z2,output_shape[0])]

        # Initialize
        result = np.full(output_shape,1)

        num_classes = 2
        if boundary_class:
                # Note the order is switched!!!
                result[zbox[0]:zbox[1],xbox[0]:xbox[1]] = 2
                num_classes+=1
        
        # Fill in the pocket but only look in the box to save time. 
        for ix in range(xbox[0],xbox[1]):
                for iz in range(zbox[0],zbox[1]):
                        x_dist = (x_node-ix)*x_step
                        z_dist = (z_node-iz)*z_step

                        dist = np.sqrt(x_dist**2+z_dist**2)
                        if dist <= radius:
                                # Note the order is switched!!!
                                result[iz,ix] = 0
        # One hot it
        result.shape = output_shape[0]*output_shape[1]
        result =np.array(list(map(lambda x: one_hot_it(x,num_classes), result)))
        result.shape = output_shape+[num_classes]

        return result

def ground_truth_2d_circle_pocket(raw_labels, 
                                  xmax = 2.0,
                                  zmax = 3.0,
                                  output_shape = [120,180], 
                                  boundary_class = False):
        """
        Expands raw labels into ground truths for 2D circle
        pocket problems. 
        
        Parameters:
        -----------
                raw_labels: A numpy array of shape (N,3) where
                        each row is a label (radius, x , z).
                xmax: The length of the x direction. (float)
                zmax: The length of the z direction. (float)
                output_shape: The shape for generated output. Note that
                        [m,n] will have the m pixels in the z direction
                        and n pixels in the x direction. 
        Returns:
        --------
                A numpy array of shape [N,m,n,3] where the ith rowp(i,j,k) = 1 
                means that location (i,j) belongs to class k (i.e. it is a one hot vector).
        """
        num_samples = raw_labels.shape[0]
        results = [0]*num_samples
        for i in range(num_samples):
                r, xc, zc = raw_labels[i,...]
                results[i] = ground_truth_2d_circle_pocket_single_example(r,xc,zc,
                                                                xmax=xmax,
                                                                zmax=zmax,
                                                                output_shape=output_shape,
                                                                boundary_class=boundary_class)
        return np.array(results)


def ground_truth_sine_single_example(ct,cb,D,f,
                                        amplitude = 0.1,
                                        xmax = 2.0,
                                        zmax = 3.0,
                                        output_shape = [120,180], 
                                        boundary_class = False):
        """
        Class map {0: low speed (pocket), 1: high speed}
        Generates a ground truth for the sine interface problem. 
        Interface function:
                F(x,z) = z-amplitude*sin(f*PI*) = D
        Since (0,0) is the top left corner of a grid (that is depth is measured
        downwards), the top region is when F < D and the bottom is when F > D. 
        
        Currently has high speed being top hardcoded. Can change this later
        if we want to switch it up
        
        Parameters:
        -----------
                ct: Wave speed of top region. (float)
                cb: Wave sepped of bottom region. (float)
                D : Center line of sine curve. (float)
                f : Frequency of sine curve. (float)
                amplitude: The amplitude of the sine curve in km. (float)
                xmax: The length of the x direction. (float)
                zmax: The length of the z direction. (float)
                output_shape: The shape for generated output. Note that
                        [m,n] will have the m pixels in the z direction
                        and n pixels in the x direction.
        Returns:
        --------
                A numpy array of shape [M,N,3] where p(i,j,k) = 1 means that 
                location (i,j) belongs to class k (i.e. it is a one hot vector).
        """
        # Total nodes, notice order swap.
        Nx,Nz = output_shape[1], output_shape[0]
        # Get a step for each index note order swap.
        x_step, z_step = xmax/float(Nx), zmax/float(Nz) # km/pixel in each direction

        # Find the centerline index.
        D_ix = int(D/float(zmax) * Nz)
        # Convert amplitude to 
        amplitude_pixels = int(amplitude/float(z_step))+1

        # Make a band around sine interface
        # to reduce number of pixels checked.
        z_band = [max(D_ix - amplitude_pixels,0), min(D_ix+amplitude_pixels,Nz)]

        # Fill in easy ones.
        result = np.zeros(output_shape+[2])
        # HARDCODED THE TOP TO BE HIGHSPEED
        result[:z_band[0],:,1] = 1
        result[z_band[1]:,:,0] = 1
        
        for ix in range(Nx):
                for iz in range(z_band[0],z_band[1]):
                        x_loc,z_loc = x_step*ix, z_step*iz 
                        val = z_loc-amplitude*np.sin(f*np.pi*x_loc)
                        if val < D:
                                result[iz,ix,1] = 1
                        else: # Assigns boundary to lowspeed region
                                result[iz,ix,0] = 1
        
        return result

def ground_truth_sine(raw_labels, 
                        amp = 0.1,
                        xmax = 2.0,
                        zmax = 3.0,
                        output_shape = [120,180]):
        """
        Expands raw labels into ground truths for sine
        interface problems.
        
        Parameters:
        -----------
                raw_labels: A numpy array of shape (N,3) where
                        each row is a label (radius, x , z).
                xmax: The length of the x direction. (float)
                zmax: The length of the z direction. (float)
                output_shape: The shape for generated output. Note that
                        [m,n] will have the m pixels in the z direction
                        and n pixels in the x direction. 
        Returns:
        --------
                A numpy array of shape [N,m,n,3] where the ith rowp(i,j,k) = 1 
                means that location (i,j) belongs to class k (i.e. it is a one hot vector).
        """
        num_samples = raw_labels.shape[0]
        results = [0]*num_samples
        for i in range(num_samples):
                ct, cb, D, f = raw_labels[i,...]
                results[i] = ground_truth_sine_single_example(ct,cb,D,f,amplitude=amp,
                                                                xmax=xmax,
                                                                zmax=zmax,
                                                                output_shape=output_shape)
        return np.array(results)
