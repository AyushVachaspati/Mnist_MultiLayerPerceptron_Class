import pandas as pd
def get_dataset():
    """ Reads Mnist dataset from csv file.
	returns normalized dataset"""
    train_set = pd.read_csv("dataset/mnist_train.csv")
    train_pixels =  train_set.iloc[:,1:].values
    train_labels= train_set.iloc[:,0].values
    print "Number of Training Samples: " + str(len(train_pixels))

    test_set = pd.read_csv("dataset/mnist_test.csv")
    test_pixels =  test_set.iloc[:,1:].values
    test_labels= test_set.iloc[:,0].values
    print "Number of Test Samples: " + str(len(test_pixels))

    return train_pixels,train_labels,test_pixels,test_labels
