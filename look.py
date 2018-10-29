import numpy as np 
import matplotlib.pyplot as plt 

raw_pred, raw_real = np.load('y_pred_test.npy'), np.load('y_actual_test.npy')
#print(raw_real.shape, raw_pred.shape)
real = np.argmax(raw_real, axis=-1)
pred = np.argmax(raw_pred, axis=-1)
pred_heat = np.max(raw_pred, axis=-1)
#pred1 = pred[:,:,0]
#pred2 = pred[:,:,1]
#pred = np.argmax(pred, axis=-1)
#real = real[:,:,0]
#pred = pred[:,:,0]

for_plt = [real, pred, pred_heat]
#print(for_plt[0].shape)
#print(for_plt[2].shape)
#for_plt = np.concatenate(for_plt, axis = 0)
#print(for_plt.shape)

aspect = 1/5

plt.subplot(221)
plt.title('Actual Labels')
plt.xlabel('Normalized Depth')
plt.ylabel('Sample')
plt.imshow(for_plt[0], aspect=aspect)
#plt.colorbar()

plt.subplot(222)
plt.title('Predicted Labels')
plt.xlabel('Normalized Depth')
plt.ylabel('Sample')
plt.imshow(for_plt[1], aspect=aspect)
#plt.colorbar()

plt.subplot(223)
plt.title('Difference')
plt.xlabel('Normalized Depth')
plt.ylabel('Sample')
plt.imshow(for_plt[1] - for_plt[0], aspect=aspect)
#plt.colorbar()

plt.subplot(224)
plt.title('Heatmap')
plt.xlabel('Normalized Depth')
plt.ylabel('Sample')
plt.imshow(for_plt[2], aspect=aspect)
plt.colorbar()

plt.suptitle('Two Layer AE with Dense Blocks - Interface Detection')
plt.show()

#plt.imshow(pred, aspect=100)
#plt.show()