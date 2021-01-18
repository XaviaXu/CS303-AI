import numpy as np

arr = np.zeros(8)
arr[1] = arr[4] = arr[5] = 8
max = np.where(arr==np.max(arr))[0]
while True:
    ind = np.random.randint(len(max))

