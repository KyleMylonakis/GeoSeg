import numpy as np
import scipy.io as sio

data_mat_contents = sio.loadmat('data_bucket.mat')
label_mat_contents = sio.loadmat('label_bucket.mat')

data_bucket = data_mat_contents['data_bucket']
#print('Max of data_bucket ', np.max(data_bucket))
label_bucket = label_mat_contents['label_bucket']
#print('Label Bucket: ', label_bucket)

#print(type(data_bucket), data_bucket.shape)
np.save('station_data_tmp',data_bucket)
np.save('label_data_tmp',label_bucket)
