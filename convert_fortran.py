import os
import argparse
import numpy as np

'''
This file converts Jim's fortran data into data and label npy files split
between training and evaulation datasets
'''

def precision_chop(ls, precs):
    """
    Converts a list strings into floats with 
    a predefined decimal placement. 
    Ex:
        (['abc','def'],[0,2]) -> [.abc,de.f]
    Parameters:
    -----------
        ls: A list of strings whose elements are integers.
        precs: A list of integers specifying where to place decimal.
    Returns:
    --------
        A list of the floated strings with the proper decimal \
        placement.
    """
    n = len(ls)
    assert n == len(precs), 'List of strings must be same length as list of precisions'
    result = [0]*n

    for i in range(n):
        val = ls[i]
        p = precs[i]

        val = val[:p]+'.'+val[p:]
        result[i] = float(val)
    return result

if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fortran-data-dir',
                        help = 'Path to the fortran data',
                        type = str,
                        required = True)
    
    parser.add_argument('--bucket-size',
                        help = 'Number of fortran files per numpy file',
                        type = int,
                        default = 20833)

    parser.add_argument('--num-annotations',
                        help = '(Deprecated) The number of annotations',
                        type = int,
                        required = False)
    
    parser.add_argument('--annotation-precisions',
                        help= 'The precision to use for each annotation term',
                        nargs = '+',
                        type = int,
                        required = True)

    parser.add_argument('--num-receivers',
                        help = 'The number of receiver stations',
                        type = int,
                        required = True)

    parser.add_argument('--num-time-steps',
                        help = 'The number of time steps in the simulation',
                        type = int,
                        required = True)

    parser.add_argument('--split',
                        help = 'The percent split between training and evaulation data.\
                            Must be and integer between 0 and 100',
                        type = int,
                        default = 80)

    parser.add_argument('--switch-recv-spatial',
                        help = 'Whether to transpose the receiver and spatial coordinates',
                        type = bool,
                        default = True)

    parser.add_argument('--save-dir',
                        help = 'Path to where the numpy files should be saved',
                        type = str,
                        required = True)

    args = parser.parse_args()
                        
    # Set parsed args as variables
    fortran_data_path = args.fortran_data_dir
    
    if args.bucket_size is None:
        bucket_size = len(os.listdir(fortran_data_path))
    else:
        bucket_size = args.bucket_size
    num_annotations = args.num_annotations
    num_receivers = args.num_receivers
    num_time_steps = args.num_time_steps
    split = args.split
    switch_recv_spatial = args.switch_recv_spatial
    save_dir = args.save_dir
    precisions = args.annotation_precisions


    # If save dir doesn't exist, make it
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Make list of every file in fortran data path and shuffle it
    fortran_data = os.listdir(fortran_data_path)
    np.random.shuffle(fortran_data)

    # Other necessary paremeters
    num_files = len(fortran_data)
    num_buckets = int(np.ceil(num_files/bucket_size))
    num_dims = 3

    # Create the label and data buckets
    baddies = 0
    bad_files = []
    for current_bucket in range(0,num_buckets):
        
        # The last bucket will in general be smaller than the rest of the buckets
        if current_bucket == num_buckets - 1:
            if num_files % bucket_size != 0:
                bucket_size = num_files % bucket_size
        
        # Initialize the current bucket
        data_bucket = np.zeros((bucket_size, num_time_steps, num_dims, num_receivers))
        label_bucket = np.zeros((bucket_size, len(precisions)))


        # Start populating the buckets with the fortran data
        start = current_bucket*bucket_size
        for ii in range(0,bucket_size):
            print("Opening ", fortran_data[start + ii])

            # Make the annotations for a given example
            name = fortran_data[start + ii]
            # The following assumes the first element of the list numbers
            # is something like Stations or SEM, if Jim changes the dataformat
            # this will need to change
            #numbers = list(map(lambda x: precision_chop(x,precisions),name.split('_')[1:] ))
            numbers = precision_chop(name.split('_')[1:],precisions)
            assert (len(numbers) == len(precisions)), 'The number of annotations present on the file does not match the number of specified annotations'
            label_bucket[ii,...] = numbers

            # Open the fortran data dump

            tmp = np.fromfile(os.path.join(fortran_data_path, fortran_data[start + ii]), dtype=np.float_)

            # Normalize the data
            assert np.isfinite(tmp), "Some data is not a finite number."
            assert (np.max(tmp) != 0), "Divide by zero error caused by data being uniformly zero."
            tmp = tmp / np.max(tmp)

            # Add data from file to the bucket using Fortran ordering
            
            try:
                data_bucket[ii,...] = tmp.reshape((num_time_steps,num_dims,num_receivers), order='F')
            except:
                baddies+=1
                err = "File {} has {} elements and cannot be reshaped to {} \n There are {} baddies"
                print(err.format(name,tmp.shape,(num_time_steps,num_dims,num_receivers),baddies))
                bad_files.append(name+',%d'%tmp.shape[0])


        
        # Transpose the spatial and receiver coordinates
        if switch_recv_spatial:
            print("Transposing the spatial and receiver coordinates")
            data_bucket = np.transpose(data_bucket, axes=(0,1,3,2))

        # Create training and eval subdirectories if necessary
        train_save_path = os.path.join(save_dir, 'train')
        eval_save_path = os.path.join(save_dir, 'eval')

        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)
        if not os.path.isdir(eval_save_path):
            os.mkdir(eval_save_path)

        train_data_path = os.path.join(train_save_path, 'data_bucket_' + str(current_bucket + 1))
        train_label_path = os.path.join(train_save_path, 'label_bucket_' + str(current_bucket + 1))
        eval_data_path = os.path.join(eval_save_path, 'data_bucket_' + str(current_bucket + 1))
        eval_label_path = os.path.join(eval_save_path, 'label_bucket_' + str(current_bucket + 1))

        # Save the buckets
        print("Saving training and evaluation data and label bucket number ", current_bucket + 1)
        split_pt = int(split * bucket_size /100 )
        np.save(train_data_path, data_bucket[0:split_pt,...])
        np.save(train_label_path, label_bucket[0:split_pt,...])
        np.save(eval_data_path, data_bucket[split_pt:,...])
        np.save(eval_label_path, label_bucket[split_pt:,...])

        with open('baddies.txt','w') as fw:
            for baddy in bad_files:
                fw.write(baddy)
        
