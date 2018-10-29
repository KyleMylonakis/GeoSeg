import numpy as np
import scipy.io as sio
import os
import hdf5storage


def save_as_np (data_filename, label_filename, train_or_eval):
    data_mat_contents = hdf5storage.loadmat(data_filename)
    label_mat_contents = hdf5storage.loadmat(label_filename)

    # Names come from matlab_script. Must be data_bucket
    # and label bucket respectively
    data_bucket = data_mat_contents['data_bucket']
    #print('Max of data_bucket ', np.max(data_bucket))
    label_bucket = label_mat_contents['label_bucket']
    #print('Label Bucket: ', label_bucket)

    #print(type(data_bucket), data_bucket.shape)
    # [0:-4] chops off '.mat' from the filename
    data_dir = 'fga_data_set/' + train_or_eval

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    np.save(data_dir + '/' + data_filename[0:-4],data_bucket)
    np.save(data_dir + '/' + label_filename[0:-4],label_bucket)

num_mat_files = 0

for name in os.listdir():
    if name.endswith('.mat'):
        num_mat_files +=1


split = 0.8
#split = 0
if( (num_mat_files//2)*split > 1):
    n_train_files = int(np.ceil(num_mat_files//2* split))

    for jj in range(0,n_train_files):
        data_filename = 'data_bucket_' + str(jj + 1) + '.mat'
        label_filename = 'label_bucket_' + str(jj + 1) + '.mat'
        save_as_np(data_filename,label_filename,'train')
    for jj in range(n_train_files,num_mat_files//2):
        data_filename = 'data_bucket_' + str(jj + 1) + '.mat'
        label_filename = 'label_bucket_' + str(jj + 1) + '.mat'
        save_as_np(data_filename,label_filename,'eval')
else:
    data_mat_contents = hdf5storage.loadmat('data_bucket_1.mat')
    label_mat_contents = hdf5storage.loadmat('label_bucket_1.mat')

    data_bucket = data_mat_contents['data_bucket']
    #print('Max of data_bucket ', np.max(data_bucket))
    label_bucket = label_mat_contents['label_bucket']
    #print('Label Bucket: ', label_bucket)
    cut = int(split*data_bucket[:,0,0,0].size)
    print("CUT AT ", cut)
    train_bucket = data_bucket[0:cut,...]
    eval_bucket = data_bucket[cut:,...]
    train_labels = label_bucket[0:cut,...]
    eval_labels = label_bucket[cut:,...]
    #print(type(data_bucket), data_bucket.shape)
    
    np.save('fga_data_set/Pwave_data_set/FGA/train/data_bucket_1',train_bucket)
    np.save('fga_data_set/Pwave_data_set/FGA/train/label_bucket_1',train_labels)
    np.save('fga_data_set/Pwave_data_set/FGA/eval/data_bucket_2',eval_bucket)
    np.save('fga_data_set/Pwave_data_set/FGA/eval/label_bucket_2',eval_labels)
    
    '''np.save('fga_data_set/Pwave_data_set/SEM/train/data_bucket_1',train_bucket)
    np.save('fga_data_set/Pwave_data_set/SEM/train/label_bucket_1',train_labels)
    np.save('fga_data_set/Pwave_data_set/SEM/eval/data_bucket_2',eval_bucket)
    np.save('fga_data_set/Pwave_data_set/SEM/eval/label_bucket_2',eval_labels)'''
    
    '''np.save('fga_data_set/SP_wave_data_set/train/data_bucket_1',train_bucket)
    np.save('fga_data_set/SP_wave_data_set/train/label_bucket_1',train_labels)
    np.save('fga_data_set/SP_wave_data_set/eval/data_bucket_2',eval_bucket)
    np.save('fga_data_set/SP_wave_data_set/eval/label_bucket_2',eval_labels)'''

    '''np.save('fga_data_set/SP_wave_data_set/SEM/train/data_bucket_1',train_bucket)
    np.save('fga_data_set/SP_wave_data_set/SEM/train/label_bucket_1',train_labels)
    np.save('fga_data_set/SP_wave_data_set/SEM/eval/data_bucket_2',eval_bucket)
    np.save('fga_data_set/SP_wave_data_set/SEM/eval/label_bucket_2',eval_labels)'''

    #raise Exception('Not enough data files. To seperate into training and evaluation data sets')
