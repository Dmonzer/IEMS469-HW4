
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#!nvidia-smi
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import  sklearn
from sklearn.model_selection import train_test_split
#import ray
import os
import math
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers.schedules as schedules
import pickle
from sklearn.metrics import f1_score

from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import logging
from tensorflow.keras.optimizers import SGD


tf.get_logger().setLevel(logging.ERROR)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        


'''
originaldir = os.getcwd()
## Change to the directory of where you keep your files.
os.chdir('/nfs/home/dem1110/Assignment 4/')
'''

train_X_all = np.load('train_X.npy', allow_pickle=True)
train_Y_all  = np.load('train_y.npy', allow_pickle=True)
test_X = np.load('test_X.npy', allow_pickle=True)
test_Y = np.load('test_y.npy', allow_pickle=True)

'''
os.chdir(originaldir)
os.getcwd()
'''
train_X_all.shape

train_Y_all.shape

test_X.shape

"""### Splitting the data into training and validation

"""

# Split the data into training and validation sets
x_train, x_val= train_test_split(
    train_X_all,
    test_size=0.2,
    random_state=42)

y_train, y_val= train_test_split(
    train_Y_all,
    test_size=0.2,
    random_state=42)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
#combined_dataset = train_dataset.concatenate(val_dataset)
x_val_tf = tf.convert_to_tensor(x_val)

type(x_train)



x_train.shape

len(train_dataset)

"""### Define model function & function that takes variables and runs the model."""

def build_model( batch_size, activation):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation),
        Dense(10, activation='softmax')
    ])
    
    '''
    custom_optimizer = SGD(learning_rate=lr )

    model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=[macro_f1])
    '''
    
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=[macro_f1])
    return model

def macro_f1(y_true, y_pred):
    # Convert softmax probabilities to class predictions
    y_pred_classes = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    y_true_classes = tf.cast(y_true, tf.int32)

    def _numpy_f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    # Use tf.py_function to wrap the NumPy function
    ##so that it can be executed within TensorFlow's computational graph
    f1 = tf.py_function(_numpy_f1_score, (y_true_classes, y_pred_classes), tf.float64)
    return f1


def run_model(activation_ind, batch_size_exp, train_dataset,val_dataset ):
    activation = ['relu', 'sigmoid', 'tanh'][int(activation_ind)]
    batch_size = int(2 ** batch_size_exp)

    E = 600
    lr = 0.01
    #train_dataset = train_dataset
    #val_dataset = val_dataset

    model = build_model(batch_size, activation)
    '''
    model.fit(train_dataset.batch(batch_size), epochs=E, validation_data=val_dataset.batch(batch_size), verbose=0)
    train_loss, train_f1 =model.evaluate(train_dataset.batch(batch_size), verbose=0)
    val_loss, val_f1 = model.evaluate(val_dataset.batch(batch_size), verbose=0)
    model.fit(train_X, train_y, epochs=E, validation_data=(val_X, val_y), verbose=0)

    '''
    num_train_samples = len(x_train)
    num_train_batches = (num_train_samples + batch_size - 1) // batch_size

    for epoch in range(E):
        epoch_loss = 0.0
        epoch_fitness = 0.0

        for i in range(num_train_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_train_samples)
            batch_X, batch_y = x_train[start_idx:end_idx], y_train[start_idx:end_idx]

            # Train on batch
            batch_loss, batch_fitness = model.train_on_batch(batch_X, batch_y)

            epoch_loss += batch_loss * len(batch_X)
            epoch_fitness += batch_fitness * len(batch_X)

        epoch_loss /= num_train_samples
        epoch_fitness /= num_train_samples

        #print(f"Epoch {epoch + 1}/{E} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_fitness:.4f}")

    # Evaluation using validation data
    val_preds = np.argmax(model.predict(x_val, verbose=0), axis=1)
    f1 = f1_score(y_val, val_preds, average='macro')



    return f1 #val_f1

def optimize_parameters(train_dataset, val_dataset):
    def parameters_model(activation_ind, batch_size_exp):

        return run_model(activation_ind, batch_size_exp, train_dataset,val_dataset )
    print("started optimizing",  flush=True)
    optimizer = BayesianOptimization(
        f=parameters_model,
        pbounds={'activation_ind': (0, 2.999), 'batch_size_exp': (4, 10)},
        random_state=54,
        verbose=2)
    optimizer.maximize(n_iter=200)
    best_params = optimizer.max['params']
    best_activation = ['relu', 'sigmoid', 'tanh'][int(best_params['activation'])]
    best_batch_size = int(2 ** best_params['batch_size'])
    print("Best activation function:", best_activation)
    print("Best batch size:", best_batch_size)

    output_filename = 'optimizer_output_run4.txt'

    with open(output_filename, 'w') as file:
        for i, res in enumerate(optimizer.res):
            file.write(f"Iteration {i}:\n\t{res}\n")
            #print("Iteration {}: \n\t{}".format(i, res))

    print("Final result:", optimizer.max)




    #print("Final result:", optimizer.max)
    
    return best_activation, best_batch_size

best_activation, best_batch_size = optimize_parameters(train_dataset, val_dataset)



E = 600
lr = 0.01
best_model = build_model(best_batch_size, best_activation)
history = best_model.fit(np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)),
                         epochs=E, batch_size=best_batch_size, verbose=1)

# plot training F1 score versus epochs
train_f1 = history.history['macro_f1']
epochs = range(1, len(train_f1) + 1)
plt.plot(epochs, train_f1, 'bo', label='Training F1')
plt.xlabel('Epochs')
plt.ylabel('Training F1 Score')
plt.legend()
plt.savefig("p2- train data fitness.png")
plt.show()


# plot training loss
train_loss = history.history['loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig("p2-train-data-loss.png")
plt.show()


# evaluate on test data
test_preds = np.argmax(best_model.predict(test_X), axis=1)
test_f1 = f1_score(test_Y, test_preds, average='macro')
print("Test F1 Score:", test_f1)

# get test loss
test_loss = best_model.evaluate(test_X, test_Y)[0]
print("Test Loss:", test_loss)










