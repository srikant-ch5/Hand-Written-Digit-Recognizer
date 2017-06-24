import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import pandas as pd

train_label = []
train_data = []

print("reading the test data")
with open('C:\Users\pailla narsi reddy\Downloads/trainSub.csv', 'r') as reader:
    reader.readline()

    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])
    #extracting the pixels
    
    print("train data read")

train_data_sub = train_data[:5000]

data = np.asarray(train_data_sub)

train_label_sub = train_label[:5000]
plt.figure(1, figsize=(3, 3))
plt.imshow(data[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

print("running a classifier")
clf = svm.SVC()
clf.fit(data,train_label_sub)

test_data = []
print("Reading the test data")
with open("C:\Users\pailla narsi reddy\Downloads/test.csv") as reader2:
    reader2.readline()
    
    for line in reader2.readlines():
        data = list(map(int,line.rstrip().split(',')))
        test_data.append(data)
    print("Test data read of size as list", len(test_data))
test_sub = test_data[:100]
test_sub = np.asarray(test_sub)
print("Test data has been converted into an array")

print(test_sub[5])
