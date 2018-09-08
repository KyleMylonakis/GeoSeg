import numpy as np
import scipy.io as sio
import os


def save_as_np (data_filename, label_filename, train_or_eval):
    data_mat_contents = sio.loadmat(data_filename)
    label_mat_contents = sio.loadmat(label_filename)

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


split = 0.7
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
    raise Exception('Not enough data files. To seperate into training and evaluation data sets')
