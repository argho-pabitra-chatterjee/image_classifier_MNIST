## importing the libraries

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix 
import time
from datetime import timedelta
#from matplotlib.pyplot import xlabel
from nltk.app.nemo_app import images
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


## now let us write a function to plot all these 9 images in a 3 by 3 grid
    ## we are writing the true and predicted classes below these images
def plot_images(images, true_label, pred_label=None):
    assert len(images) == len(true_label) == 12
    
    ## create figure with 3 by 3 sub-plot
    fig, axes = plt.subplots(3,4)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i].reshape(img_shape),cmap = 'binary')
        
        ## showing the true and predicted classes
        if pred_label is None:
            x_label = "True: {0}".format(true_label[i])
        else:
            x_label = "True: {0}, Pred: {1}".format(true_label[i],pred_label[i])
        
        ## show the classes as the label on the x-axis
        ax.set_xlabel(x_label)
        
        # remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
        
    ## showing the plot
    plt.show()
            
## new TF variables in the given shape and initializing them with some random values
## the initialization is merely being defined in the TF graph            
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    # equivalent to y-intercept
    # constant value carried over across matrix math
    # a baise across multidimensional array in a CNN helps in convergence
    # it helps to make our model more accurate 
    return tf.Variable(tf.constant(0.05, shape = [length]))

## defining a new Convolutional layer
##
## 4-dimensional input:
##  1) Image number.
##  2) Y-axis of each image.
##  3) X-axis of each image.
##  4) Channels of each image.
##
## 4-dimensional output:
##  1) Image number, same as input.
##  2) Y-axis of each image. If 2x2 pooling is used, then the height and width of the input images is divided by 2.
##  3) X-axis of each image. Ditto.
##  4) Channels produced by the convolutional filters.
## 
## It has pooling and ReLU built in

def new_conv_layer(
            input_prev_layer,   ## coming from previous layer
            num_input_channels, ## Number of channels in previous layer
            filter_size,        ## width and height of each filter
            num_filters,        ## number of filters
            use_pooling = True  ## use a 2 X 2 max pooling
        ):
            shape = [filter_size, filter_size, num_input_channels, num_filters]
            
            # create new filters with the given shape
            weights = new_weights(shape)
            
            # create new biase for each filter
            biases = new_biases(length = num_filters)
            
            # creating a CNN layer
            layer = tf.nn.conv2d(
                    input = input_prev_layer,
                    filter = weights,
                    strides = [1,1,1,1],         # determines by how much the convolution window will move
                    padding = 'SAME'
                )
    
            # adding the baise to the layer
            layer = layer + biases
                
            ## we can use pooling to downsample the image resolution
            if use_pooling:
                # we do a 2 X 2 pooling, where we select the largest value in the window
                layer = tf.nn.max_pool(
                    value = layer, 
                    ksize = [1,2,2,1], 
                    strides = [1,2,2,1], 
                    padding = 'SAME'
                    )
                
            ## ReLU - Rectified Linear Unit. 
            ## makes negative input_prev_layer-pixel as 0
            ## This adds non-linearity to the formula and allows to learn more complicated features.
            layer = tf.nn.relu(layer)
            
            # Note that ReLU is normally executed before the pooling,
            # but since relu(max_pool(x)) == max_pool(relu(x)) we can
            # save 75% of the relu-operations by max-pooling first.

            # We return both the resulting layer and the filter-weights
            # because we will plot the weights later.
            return layer, weights
    


## defining the fully connected layer.
## we add a fully connected layer after convolution layer, there by we reduce our 4-D array to 2-D array
## to be used as input_prev_layer to this layer    
def flatten_layer(layer):
    # shape of input layer
    layer_shape = layer.get_shape()
    # the shape of the input layer is [num_images, img_height, img_width, num_channels]
    
    # the number of features = img_height X img_width X num_channels
    # We calculate this using a function of tensor flow
    num_features = layer_shape[1:4].num_elements()
    
    # we reshape this layer from 4-D input to 2-D output
    # output -> num_images, num_features
    # [num_images, img_height, img_width, num_channels]  ->  [num_images, num_features]
    # num_features = img_height X img_width X num_channels
    # num_images = -1 [the size of the first dimension is set to -1]
    layer_flat = tf.reshape(layer, [-1, num_features])
    
    ## shape of flattened layer  = [num_images, img_height * img_width * num_channels]
    return layer_flat, num_features
    

## Function to create a new fully connected layer.
## The input is 2-D tensor and output is 2-D tensor
def new_fc_layer(
        input_prev_layer,           # input_prev_layer from previous layer
        num_inputs,                 # no. of inputs from previous layer
        num_outputs,                # no. of outputs from this layer
        use_relu = True             # Use Rectified Linear Unit (ReLU)?
        ):
    ## creating weights and biases for this layer
    weights = new_weights(shape = [num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    
    # computing the layer by multiplying Input with weights and adding biases
    layer = tf.matmul(input_prev_layer,weights) + biases
    
    # applying ReLU : non-linear transformation
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer

    
## Function for performing a number of optimization iterations so as 
## to gradually improve the variables of the network layers.
## Counter for total number of iterations performed so far.
total_itr = 0

def optimize_itr(num_itr, optimizer):
    # Ensure we update the global variable rather than a local copy.
    global total_itr
    
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_itr,
                   total_itr + num_itr):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_itr += num_itr

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
  
## plot misclassified images with their true class labels
def plot_example_errors(pred_label, correct):
    # This function is called from print_accuracy() below.

    # pred_label is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    pred_label = pred_label[incorrect]

    # Get the true classes for those images.
    true_label = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                true_label=true_label[0:9],
                pred_label=pred_label[0:9])
    

## plot the confusion matrix

def plot_confusion_matrix(pred_label):
    # This is called from print_accuracy() below.

    # pred_label is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    true_label = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=true_label,
                          y_pred=pred_label)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()   
 
 
# Split the test-set into smaller batches of this size.
# In case of lower RAM the below value needs to be lowered
#test_batch_size = 256
test_batch_size = 256

def print_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    pred_label = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        pred_label[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    true_label = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (true_label == pred_label)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(pred_label=pred_label, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(pred_label=pred_label)


 
 
### execution of the main function
### starting point of the code    
if __name__ == '__main__':
    # Convolutional layer 1.
    filter_size1 = 5                     # Convolutional size filter are 5 X 5 filters
    num_filters1 = 16                   # there are 16 of these filters
    
    # fully-connected layer.
    fc_size = 128                       # number of nuerons in fully connected layers
    
    # convolutional layer 2
    filter_size2 = 5
    num_filters2  = 36
    
    print('reading/downloading dataset..')
    
    ## read the dataset
    data = input_data.read_data_sets('../data/MNIST', one_hot = True)
    
    print('size of the training dataset : ' + str(len(data.train.labels)))
    print('size of the training dataset : ' + str(len(data.test.labels)))
    print('size of the training dataset : ' + str(len(data.validation.labels)))

    data.test.cls = np.argmax(data.test.labels, axis = 1)
    
    
    ## defining the dimension of the data
    ## MNIST images are all 28 pixels by 28 pixels
    img_size = 28
    
    ## These 28 by 28 pixel image is stored in a 1-D array
    img_size_flat = img_size * img_size
    
    ## image shape - height and width of image to reshape arrays
    img_shape = (img_size, img_size)
    
    ## number of color channels for the image - Only gray scale image hence we use 1 channel
    num_channels = 1
    ## in case of color we use number of channels as 3 - RGB
    
    ## number of classes
    num_classes = 10    ## each class represents one digit[0-9]
    
    ## checking few images if the data is correct
    ## getting the 1st image from train dataset
    train_frst_img = data.test.images[0:12]
    # getting the true class for the image
    true_label = data.test.cls[0:12]
    #plotting the image and labeling using helper function
    plot_images(images = train_frst_img, true_label = true_label)
    
    # defining the placeholder variables
    # The below tensor just means that it is a multi-dimensional vector or matrix. 
    # The data-type is set to float32 and the shape is set to [None, img_size_flat], 
    # where 'None' means that the tensor may hold an arbitrary number of images 
    # with each image being a vector of length img_size_flat.
    x = tf.placeholder(tf.float32, shape = [None, img_size_flat], name='x')
    
    # The convolution layer expects image to be in 4-D tensor, so let us reshape it to 4-D
    # 4-D tensor = [num_images, img_height, img_width, num_channels]
    # In this case,  img_height = img_width = img_size; num_images can be inferred using -1 as size of dimension
    x_image = tf.reshape(tensor=x, shape=[-1, img_size, img_size, num_channels])
        
    # Next, we define the placeholder of 'y' variable, which is true label of the image
    # The shape of this placeholder variable 
    y_true = tf.placeholder(tf.float32, shape = [None, 10], name = 'y_true')
    
    
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    # creating the 1st Convolutional layer
    # We are down scaling the image to (?, 14, 14, 16)
    layer_conv1, weights_conv1 = new_conv_layer(input_prev_layer=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
    
    layer_conv1
    
    # creating the 2nd Convolutional layer
    layer_conv2, weights_conv2 = \
    new_conv_layer(input_prev_layer=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True) 
    # We are down scaling the image to (?, 7, 7, 36)
    layer_conv2

    # We want the input from convolutional layer to be flattend so that we can give it to the fully connected layer
    layer_flat, num_features = flatten_layer(layer_conv2)
    # flattened input = (?, 1764) where 1764  = 7 x 7 x 36 i.e. number of features     
    layer_flat

    # Creating the first fully connected layer
    # The number of neurons or nodes in the fully-connected layer is fc_size
    layer_fc1 = new_fc_layer(input_prev_layer=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
    
    # Output of a fully connected layer  = (?, 128)
    layer_fc1
    
    
    # another fully connected layer to determine which of the 10 classes the image belongs to 
    # ReLU is not used for this layer
    layer_fc2 = new_fc_layer(input_prev_layer=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
    
    # shape of the output = (?, 10)
    layer_fc2
    
    # predicted class
    y_pred = tf.nn.softmax(layer_fc2)

    # The index for the largest element.
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
    
    cost = tf.reduce_mean(cross_entropy)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    
    
    # calculates the classification accuracy by first type-casting the vector of booleans to floats, 
    # so that False becomes 0 and True becomes 1, and then calculating the average of these numbers.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # creating tensorflow session to execute the above created graph [CNN layer model]
    session = tf.Session()
    
    session.run(tf.global_variables_initializer())
    
    # There are 55000 images in training. We use small batch of image in each Iteration to Optimize
    train_batch_size = 64
    
    
    ## accuracy is low
    print_accuracy(show_confusion_matrix=True)
    
    
    optimize_itr(num_itr=1, optimizer = optimizer)
    print_accuracy(show_confusion_matrix=True)
    
    optimize_itr(num_itr=99, optimizer = optimizer)
    print_accuracy(show_confusion_matrix=True)
    
    optimize_itr(num_itr=1000, optimizer = optimizer)
    print_accuracy(show_confusion_matrix=True)
