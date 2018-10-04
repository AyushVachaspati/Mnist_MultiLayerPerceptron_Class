import numpy as np
import tensorflow as tf

class Mnist:
    num_pixels = 28*28
    batch_size = 16
    num_labels = 10
    layers_cells = [800,]
    sess = tf.Session()
    
    #model global variables which are required to give input and get output
    minimize = None
    y_pred = None
    pixels = tf.placeholder(dtype = tf.float32, shape = [batch_size, num_pixels])
    labels = tf.placeholder(dtype = tf.float32, shape = [batch_size,num_labels])
    
    def __init__(self,num_pixels=28*28,batch_size=16,num_labels=10,layers_cells=[800]):
        """num_pixels : number of input features/pixels in the image. default 28x28.
            batch_size : size of batch for mini batch gradient descent. default 16
            num_labels : number of digits to be recognized. default 10
            layer_cells : list of number of cells in each layer of MLP. defult [800,]"""
        
        self.num_pixels = num_pixels
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.layers_cells = layers_cells
    
    def build_graph(self,):
        """This function builds the graph for the model and initializes the variables to be trained"""
        
        network = [self.pixels]
        with tf.variable_scope("hidden_layers"):
            for i in range(len(self.layers_cells)):
                temp = tf.contrib.layers.fully_connected(network[i], 
							num_outputs = self.layers_cells[i],
							activation_fn=tf.nn.relu)
                network.append(temp)
        
        with tf.variable_scope("output_layer"):
            output = tf.contrib.layers.fully_connected(network[-1], 
						       num_outputs = self.num_labels, 
						       activation_fn=tf.nn.softmax)
        
        with tf.variable_scope("prediction"):
            self.y_pred = tf.argmax(output,axis = 1)
        
        with tf.variable_scope("loss"):
            cross_entropy = tf.losses.softmax_cross_entropy(self.labels,output)
        
                
        learning_rate = 5e-4  
        self.minimize = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
    def save_model(self):
        """This function saves the variable states for the trained model in order to restore it later.
           Directory saved_model must exist"""
        saver = tf.train.Saver()
        path = saver.save(self.sess,"./saved_model/model.ckpt")
        print "Model saved @ " + path
        return path
    
    def restore_model(self):
        """This function restores a previously saved model
           from saved_model directory"""
        saver = tf.train.Saver()
        saver.restore(self.sess, "./saved_model/model.ckpt")
    
    
    def get_one_hot_labels(self,labels):
        """labels: batch size number of labels"""
        encoded_labels = np.zeros([self.batch_size, self.num_labels])
        for i in range(len(labels)):
            encoded_labels[i][labels[i]] = 1
        return encoded_labels
    
    def get_accuracy(self,test_pixels, test_labels):
        """This function tests the accuracy of the current model on the test set.
            test_pixels: matrix of pixels values for the test set
            test_labels: the labels corresponding to the pixels"""
        no_of_batches = len(test_pixels)/self.batch_size
        ptr = 0
        total = 0
        correct = 0
        for i in range(no_of_batches):
            x = test_pixels[ptr:ptr+self.batch_size]
            y = self.get_one_hot_labels(test_labels[ptr:ptr+self.batch_size])
            pred_labels = self.sess.run(self.y_pred,{self.pixels: x, self.labels: y})

            correct += sum(pred_labels == test_labels[ptr:ptr+self.batch_size])
            total += len(pred_labels)

            ptr += self.batch_size

        return (correct*100.0)/total
    
    def predict_labels(self, pixels):
        """Takes a matrix of pixels for which the labels are to be predicted and returns the
            predicted labels for the same as an numpy array"""
        pred_labels = []
        #normalization of input features
        pixels = (pixels - pixels.min())/float(pixels.max()-pixels.min())
        
        no_of_batches = len(pixels)/self.batch_size
        
        ptr = 0
        for i in range(no_of_batches):
            x = pixels[ptr:ptr+self.batch_size]
            temp = self.sess.run(self.y_pred,{self.pixels: x})
            pred_labels += list(temp)
            ptr += self.batch_size
        
        x = pixels[ptr : ]
        for i in range(self.batch_size - len(x)):
            x = np.append(x,[np.zeros((self.num_pixels))],axis=0)
        temp = self.sess.run(self.y_pred,{self.pixels: x})
        pred_labels += list(temp)
        return np.array(pred_labels[:len(pixels)])
    

    def train(self,epoch,train_pixels,train_labels,test_pixels,test_labels):
        #normalize the pixel data
        train_pixels = (train_pixels - train_pixels.min())/float(train_pixels.max()-train_pixels.min())
        test_pixels = (test_pixels - test_pixels.min())/float(test_pixels.max()-test_pixels.min())
        folder_id = "1BcFfaaQ2dSjkRb-Vn1zwROMAHB_eGl-h"  #Mnist folder in google drive
        filename = "saved_model.tar.gz"
        curr_accuracy = 0
        saved_accuracy = 0
        for i in range(epoch):
            no_of_batches = len(train_pixels)/self.batch_size
            ptr = 0
            for j in range(no_of_batches):
                if j%1000 == 0 :
                    print j
                x = train_pixels[ptr:ptr+self.batch_size]
                y = self.get_one_hot_labels(train_labels[ptr:ptr+self.batch_size])
                self.sess.run(self.minimize,{self.pixels: x, self.labels: y})

                ptr+= self.batch_size
            curr_accuracy = self.get_accuracy(test_pixels,test_labels)
            print "accuracy = " + str(curr_accuracy)
            if curr_accuracy >= saved_accuracy:
                self.save_model()
                saved_accuracy= curr_accuracy
