import numpy as np 
import matplotlib.pyplot as plt 

pred, real = np.load('y_pred_test.npy'), np.load('y_actual_test.npy')
#print(real.shape, pred.shape)
real = np.argmax(real, axis=-1)
pred = np.max(pred, axis=-1)
#pred1 = pred[:,:,0]
#pred2 = pred[:,:,1]
#pred = np.argmax(pred, axis=-1)
#real = real[:,:,0]
#pred = pred[:,:,0]

for_plt = [real,pred]
#for_plt = np.concatenate(for_plt, axis = 0)
#print(for_plt.shape)

plt.subplot(221)
plt.title('Actual Labels')
plt.xlabel('Downscaled Depth')
plt.ylabel('Sample')
plt.imshow(for_plt[0], aspect=1/5)
#plt.colorbar()

plt.subplot(222)
plt.title('Predicted Labels')
plt.xlabel('Downscaled Depth')
plt.ylabel('Sample')
plt.imshow(for_plt[1], aspect=1/5)
#plt.colorbar()

plt.subplot(223)
plt.title('Difference')
plt.xlabel('Downscaled Depth')
plt.ylabel('Sample')
plt.imshow(for_plt[1] - for_plt[0], aspect=1/5)
#plt.colorbar()

plt.suptitle('UNet Interface Detection')
plt.show()

#plt.imshow(pred, aspect=100)
#plt.show()