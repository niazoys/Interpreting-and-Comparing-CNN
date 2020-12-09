import numpy as np

arr=np.load('test.npy')

arr_new=np.zeros((100,arr[0].shape[1],arr[0].shape[2],arr[0].shape[3]))

for i in range(len(arr)):
    for j in range (len(arr[0].shape[0])):
        arr_new[i*]


print(arr.shape)