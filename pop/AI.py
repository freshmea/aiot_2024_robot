import time, os, sys
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import layers, Sequential, optimizers, callbacks, losses
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, add
from tensorflow.keras import backend as K

IDENTIFIER="_models"

mnist_train_x=None
mnist_train_y=None
mnist_test_x=None
mnist_test_y=None

class Linear_Regression:
    learning_rate=1e-2
    print_every=10
    global_step=0

    _X_data=None
    _Y_data=None
    bias=None
    weight=None

    hypothesis=None

    optimizer=Adam(lr=learning_rate)

    ckpt_name="linear_regression" + IDENTIFIER
    restore=None

    _cb_fit=callbacks.Callback()

    @property
    def X_data(self):
        return self._X_data

    @X_data.setter
    def X_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2:
            arr=np.expand_dims(arr, -1)

        self._X_data=arr

    @property
    def Y_data(self):
        return self._Y_data

    @Y_data.setter
    def Y_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2:
            arr=np.expand_dims(arr, -1)

        self._Y_data=arr

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.print_every == 0:
                print(self.global_step + 1, "step loss:", logs['loss'])
        self.global_step+=1

    def __init__(self, restore=False, ckpt_name=ckpt_name):
        self.ckpt_name = ckpt_name

        self.restore = restore

        self._cb_fit.on_epoch_end=self.on_epoch_end

        self.hypothesis = Sequential()
        
        self.hypothesis.add(layers.Dense(1, input_shape=[1]))

        self.hypothesis.compile(optimizer=self.optimizer, loss='mse')

        if restore:
            if self.load(self.ckpt_name) is None:
                print("Create a new model.")
            

    def load(self, path=None):
        if path is None : path=self.ckpt_name

        if os.path.isfile(path+".index"):
            ckpt=path
            self.hypothesis.load_weights(ckpt)
            if ".ckpt-" in ckpt :
                self.global_step=int(ckpt.split(".ckpt-")[-1])
            else:
                print("[Warning] Can't read step log.")
                self.global_step=0

        elif os.path.isfile(path):
            exname = path.split(".")[-1]
            if exname == "index" or "data-" in exname :
                ckpt=path[:len(path)-(len(exname)+1)]
                self.hypothesis.load_weights(ckpt)
                if "model.ckpt-" in ckpt :
                    self.global_step=int(ckpt.split("model.ckpt-")[1])
                else:
                    print("[Warning] Can't read step log.")
                    self.global_step=0
            else:
                print("[Error] Can't find a model in '"+path+"'.")
                return

        elif os.path.isdir(path):
            if os.path.isfile(path+"/checkpoint"):
                ckpt=tf.train.latest_checkpoint(path)
                self.hypothesis.load_weights(ckpt)
                if "model.ckpt-" in ckpt :
                    self.global_step=int(ckpt.split("model.ckpt-")[1])
                else:
                    print("[Warning] Can't read step log.")
                    self.global_step=0
            else:
                print("[Error] Doesn't exist a checkpoint file in '"+path+"'.")
                return

        else:
            print("[Error] Can't find a model in '"+path+"'.")
            return

        self.ckpt_name = path
        return "Loaded successfully."

    def save(self, path=None):
        if path is None : path=self.ckpt_name

        if self.hypothesis is not None: 
            self.hypothesis.save_weights(path+"/model.ckpt-"+str(self.global_step))

    def train(self, times=100, print_every=10):
        self.print_every=print_every

        dtime=time.time()
        
        if self.X_data is not None and self.Y_data is not None :
            X_data = tf.keras.preprocessing.sequence.pad_sequences(self.X_data).astype(np.float32)
            Y_data = tf.keras.preprocessing.sequence.pad_sequences(self.Y_data).astype(np.float32)
            self.hypothesis.fit(X_data, Y_data, epochs=times, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        elif self.X_data is None :
            print("Please input a data to X_data.")
            return
        elif self.Y_data is None :
            print("Please input a data to Y_data.")
            return

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.save()

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        inputs=np.array(inputs, dtype=np.float32)
        while len(inputs.shape) < 2:
            inputs=np.expand_dims(inputs, -1)

        if self.hypothesis is not None and inputs is not None :
            return self.hypothesis.predict(inputs, use_multiprocessing=True)


class Logistic_Regression(Linear_Regression):
    ckpt_name="logistic_regression" + IDENTIFIER
    learning_rate=1e-1
    optimizer=Adam(lr=learning_rate)

    def __init__(self, input_size=1, restore=False, ckpt_name=ckpt_name):
        self.ckpt_name = ckpt_name
        self.restore = restore
        self._cb_fit.on_epoch_end=self.on_epoch_end

        self.hypothesis = Sequential()

        self.hypothesis.add(layers.Dense(1, input_shape=[input_size], activation='sigmoid'))

        self.hypothesis.compile(optimizer=self.optimizer, loss='binary_crossentropy')

        if restore:
            if self.load(self.ckpt_name) is None:
                print("Create a new model.")

    def loss_function(self, y_true, y_pred):
        y_true=tf.cast(y_true,tf.float32)
        y_pred=tf.cast(y_pred,tf.float32)
        return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * (tf.math.log(1 - y_pred)),-1)

class Perceptron:
    learning_rate=1e-2
    print_every=10
    global_step=0

    softmax=True

    _X_data=None
    _Y_data=None
    bias=None
    weight=None
    
    model=None
    _mnist_loaded=False

    optimizer=Adam(lr=learning_rate)

    ckpt_name="perceptron" + IDENTIFIER
    restore=None

    _cb_fit=callbacks.Callback()

    @property
    def X_data(self):
        return self._X_data

    @X_data.setter
    def X_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2:
            arr=np.expand_dims(arr, -1)

        self._X_data=arr

    @property
    def Y_data(self):
        return self._Y_data

    @Y_data.setter
    def Y_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2:
            arr=np.expand_dims(arr, -1)

        self._Y_data=arr

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.print_every == 0:
                print(self.global_step + 1, "step loss:", logs['loss'])
        self.global_step+=1

    def __init__(self, input_size, output_size=1, restore=False, ckpt_name=ckpt_name, softmax=True):
        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end

        self.model = Sequential()

        if self.softmax :
            if output_size > 1 :
                self.model.add(layers.Dense(output_size,input_shape=[input_size], activation='softmax'))

                self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
            else :
                self.model.add(layers.Dense(output_size,input_shape=[input_size], activation='sigmoid'))
                
                self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        else :
            self.model.add(layers.Dense(output_size,input_shape=[input_size]))
            
            self.model.compile(optimizer=self.optimizer, loss='mse')

        if restore:
            if self.load(self.ckpt_name) is None:
                print("Create a new model.")


    def load(self, path=None):
        if path is None : path=self.ckpt_name

        if os.path.isfile(path+".index"):
            ckpt=path
            self.model.load_weights(ckpt)
            if ".ckpt-" in ckpt :
                self.global_step=int(ckpt.split(".ckpt-")[-1])
            else:
                print("[Warning] Can't read step log.")
                self.global_step=0

        elif os.path.isfile(path):
            exname = path.split(".")[-1]
            if exname == "index" or "data-" in exname :
                ckpt=path[:len(path)-(len(exname)+1)]
                self.model.load_weights(ckpt)
                if "model.ckpt-" in ckpt :
                    self.global_step=int(ckpt.split("model.ckpt-")[1])
                else:
                    print("[Warning] Can't read step log.")
                    self.global_step=0
            else:
                print("[Error] Can't find a model in '"+path+"'.")
                return

        elif os.path.isdir(path):
            if os.path.isfile(path+"/checkpoint"):
                ckpt=tf.train.latest_checkpoint(path)
                self.model.load_weights(ckpt)
                if "model.ckpt-" in ckpt :
                    self.global_step=int(ckpt.split("model.ckpt-")[1])
                else:
                    print("[Warning] Can't read step log.")
                    self.global_step=0
            else:
                print("[Error] Doesn't exist a checkpoint file in '"+path+"'.")
                return

        else:
            print("[Error] Can't find a model in '"+path+"'.")
            return

        self.ckpt_name = path
        return "Loaded successfully."

    def save(self, path=None):
        if path is None : path=self.ckpt_name
        
        if self.model is not None: 
            self.model.save_weights(path+"/model.ckpt-"+str(self.global_step))

    def train(self, times=100, print_every=10):
        self.print_every=print_every

        dtime=time.time()
        
        if self.X_data is not None and self.Y_data is not None :
            X_data = tf.keras.preprocessing.sequence.pad_sequences(self.X_data).astype(np.float32)
            Y_data = tf.keras.preprocessing.sequence.pad_sequences(self.Y_data).astype(np.float32)
            self.model.fit(X_data, Y_data, epochs=times, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        elif self.X_data is None :
            print("Please input a data to X_data.")
            return
        elif self.Y_data is None :
            print("Please input a data to Y_data.")
            return

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.save()

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        inputs=np.array(inputs, dtype=np.float32)
        while len(inputs.shape) < 2:
            inputs=np.expand_dims(inputs, -1)

        if self.model is not None and inputs is not None :
            return self.model.predict(inputs, use_multiprocessing=True)

class ANN(Perceptron):
    layer=None
    ckpt_name="ANN" + IDENTIFIER

    def __init__(self, input_size, hidden_size=10, output_size=1, restore=False, ckpt_name=ckpt_name, softmax=True):
        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end

        self.model = Sequential()

        if self.softmax :
            if output_size > 1 :
                self.model.add(layers.Dense(hidden_size,input_shape=[input_size]))
                self.model.add(layers.Dense(output_size, activation='softmax'))

                self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
            else :
                self.model.add(layers.Dense(hidden_size,input_shape=[input_size]))
                self.model.add(layers.Dense(output_size, activation='sigmoid'))
                
                self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        else :
            self.model.add(layers.Dense(hidden_size,input_shape=[input_size]))
            self.model.add(layers.Dense(output_size))
            
            self.model.compile(optimizer=self.optimizer, loss='mse')

        if restore:
            if self.load(self.ckpt_name) is None:
                print("Create a new model.")

class DNN(ANN):
    learning_rate=1e-2
    level=0
    ckpt_name="DNN" + IDENTIFIER
    optimizer=Adam(lr=learning_rate)

    def __init__(self, input_size, hidden_size=10, output_size=1, layer_level=3, restore=False, ckpt_name=ckpt_name, softmax=True):
        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end

        if layer_level < 1:
            print("Please set a layer level at least 1.")
            del self
        else:
            self.model = Sequential()

            self.model.add(layers.Input(shape=(input_size,)))
            
            for _ in range(layer_level):
                self.model.add(layers.Dense(hidden_size))

            if self.softmax :
                if output_size > 1 :
                    self.model.add(layers.Dense(output_size, activation='softmax'))

                    self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
                else :
                    self.model.add(layers.Dense(output_size, activation='sigmoid'))
                    
                    self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
            else :
                self.model.add(layers.Dense(output_size))
                
                self.model.compile(optimizer=self.optimizer, loss='mse')

            if restore:
                if self.load(self.ckpt_name) is None:
                    print("Create a new model.")

class CNN(DNN):
    ckpt_name="CNN" + IDENTIFIER
    _input_level=1
    mnist_train_x=None
    mnist_train_y=None
    mnist_test_x=None
    mnist_test_y=None
    _X=None
    _Y=None
    _pre_batch_pos=0
    gray=True

    @property
    def X_data(self):
        return self._X_data

    @X_data.setter
    def X_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 3:
            if len(arr.shape) < 2 :
                arr=np.expand_dims(arr, -1)
            else :
                arr=np.expand_dims(arr, 0)

        self._X_data=arr

    @property
    def Y_data(self):
        return self._Y_data

    @Y_data.setter
    def Y_data(self, arg):
        arr=np.array(arg)
        while len(arr.shape) < 2 :
            arr=np.expand_dims(arr, -1)

        self._Y_data=arr

    def __init__(self, input_size=[28,28], input_level=1, kernel_size=[3,3], kernel_count=32, strides=[1,1], hidden_size=128, output_size=1, conv_level=2, layer_level=1, restore=False, ckpt_name=ckpt_name, softmax=True):
        self.ckpt_name = ckpt_name

        self._ipsize=input_size
        self._opsize=output_size
        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end

        input_shape=[input_size[0], input_size[1], input_level]

        if input_level == 3 :
            self.gray=False
        
        if layer_level < 1 :
            print("Please set a Fully-connected layer level at least 1.")
            del self
        elif conv_level < 1 :
            print("Please set a Convolutional layer level at least 1.")
            del self
        else:
            self.model = Sequential()

            self.model.add(layers.Input(shape=input_shape))
            
            for _ in range(conv_level):
                self.model.add(layers.Conv2D(filters=kernel_count, kernel_size=kernel_size, strides=strides, padding='SAME'))
                self.model.add(layers.MaxPool2D([2,2], [2,2], padding='SAME'))
                self.model.add(layers.Dropout(0.5))

            self.model.add(layers.Flatten())

            for _ in range(layer_level):
                self.model.add(layers.Dense(hidden_size))
                self.model.add(layers.Dropout(0.3))

            if self.softmax :
                if output_size > 1 :
                    self.model.add(layers.Dense(output_size, activation='softmax'))

                    self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
                else :
                    self.model.add(layers.Dense(output_size, activation='sigmoid'))
                    
                    self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
            else :
                self.model.add(layers.Dense(output_size))
                
                self.model.compile(optimizer=self.optimizer, loss='mse')

            if restore:
                if self.load(self.ckpt_name) is None:
                    print("Create a new model.")

    def train(self, times=100, batch=500, print_every=10):
        self.print_every=print_every

        dtime=time.time()
        
        if self.X_data is not None and self.Y_data is not None :
            X_data = tf.keras.preprocessing.sequence.pad_sequences(self.X_data).astype(np.float32)
            Y_data = tf.keras.preprocessing.sequence.pad_sequences(self.Y_data).astype(np.float32)
            self.model.fit(X_data, Y_data, epochs=times, batch_size=batch, steps_per_epoch=1, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        elif self.X_data is None :
            print("Please input a data to X_data.")
            return
        elif self.Y_data is None :
            print("Please input a data to Y_data.")
            return

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.save()

    def run(self,inputs=None,show=True):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        inputs=np.array(inputs, dtype=np.float32)
        while len(inputs.shape) < 3:
            if len(inputs.shape) < 2 :
                inputs=np.expand_dims(inputs, -1)
            else :
                inputs=np.expand_dims(inputs, 0)

        warned_shape=False

        for input in inputs:
            while len(tf.shape(input)) < 3:
                if not warned_shape:
                    print("[Warning] Inputs shape doesn't match. Automatically transformed to 4 Dimensions but may be occur errors or delay.")
                    warned_shape=True
                tf.expand_dims(input,0)

        if len(inputs)<5 and show:
            for input in inputs:
                if self.gray:
                    plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]), cmap='gray', vmin=0, vmax=255)
                else:
                    plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]))
                plt.show()

        inputs=tf.cast(inputs,tf.float32)

        if self.model is not None and inputs is not None :
            return self.model.predict(inputs, use_multiprocessing=True)

    def load_MNIST(self):
        global mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y
        if mnist_train_x is None and mnist_train_y is None:
            (self.mnist_train_x,self.mnist_train_y), (self.mnist_test_x, self.mnist_test_y) = tf.keras.datasets.mnist.load_data()
            self.mnist_train_x=self.mnist_train_x.reshape(-1, 28, 28, 1)
            self.mnist_train_y=tf.one_hot(self.mnist_train_y, 10)
            self.mnist_test_x=self.mnist_test_x.reshape(-1, 28, 28, 1)
            self.mnist_test_y=tf.one_hot(self.mnist_test_y, 10)
            self.X_data=self.mnist_train_x
            self.Y_data=self.mnist_train_y
        else:
            self.mnist_train_x=mnist_train_x
            self.mnist_train_y=mnist_train_y
            self.mnist_test_x=mnist_test_x
            self.mnist_test_y=mnist_test_y
            self.mnist_train_x=self.mnist_train_x.reshape(-1, 28, 28, 1)
            self.mnist_train_y=tf.one_hot(self.mnist_train_y, 10)
            self.mnist_test_x=self.mnist_test_x.reshape(-1, 28, 28, 1)
            self.mnist_test_y=tf.one_hot(self.mnist_test_y, 10)
            self.X_data=self.mnist_train_x
            self.Y_data=self.mnist_train_y

    def show_img(self, input):
        if self.gray:
            plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]), cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]))
        plt.show()

class RNN(ANN):
    learning_rate=1e-2
    level=0
    ckpt_name="RNN" + IDENTIFIER
    optimizer=Adam(lr=learning_rate)

    def __init__(self, hidden_size=64, output_size=1, layer_level=1, restore=False, ckpt_name=ckpt_name, softmax=True):
        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_epoch_end=self.on_epoch_end
    
        if layer_level < 1:
            print("Please set a layer level at least 1.")
            del self
        else:
            self.model = Sequential()

            self.model.add(layers.Input(shape=(None,1)))
            
            self.model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
            self.model.add(layers.MaxPooling1D(pool_size=2))
            #self.model.add(layers.Dropout(0.3))

            #self.model.add(layers.Conv1D(filters=16, kernel_size=11, strides=1, padding='valid', activation='relu'))
            #self.model.add(layers.MaxPooling1D(pool_size=3))
            #self.model.add(layers.Dropout(0.3))

            #self.model.add(layers.Conv1D(filters=32, kernel_size=9, strides=1, padding='valid', activation='relu'))
            #self.model.add(layers.MaxPooling1D(pool_size=3))
            #self.model.add(layers.Dropout(0.3))

            for _ in range(layer_level-1):
                #self.model.add(layers.Bidirectional(layers.GRU(hidden_size, return_sequences = True), merge_mode='sum'))
                self.model.add(layers.LSTM(hidden_size, return_sequences = True))

            #self.model.add(layers.Bidirectional(layers.GRU(hidden_size, return_sequences = False), merge_mode='sum'))
            self.model.add(layers.LSTM(hidden_size, return_sequences = False))

            if self.softmax :
                if output_size > 1 :
                    self.model.add(layers.Dense(output_size, activation='softmax'))

                    self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
                else :
                    self.model.add(layers.Dense(output_size, activation='sigmoid'))
                    
                    self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
            else :
                self.model.add(layers.Dense(output_size))
                
                self.model.compile(optimizer=self.optimizer, loss='mse')

            if restore:
                if self.load(self.ckpt_name) is None:
                    print("Create a new model.")

    def train(self, times=100, batch=50, print_every=10):
        self.print_every=print_every

        dtime=time.time()
        
        if self.X_data is not None and self.Y_data is not None :
            X_data = tf.keras.preprocessing.sequence.pad_sequences(self.X_data).astype(np.float32)
            Y_data = tf.keras.preprocessing.sequence.pad_sequences(self.Y_data).astype(np.float32)
            self.model.fit(X_data, Y_data, epochs=times, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        elif self.X_data is None :
            print("Please input a data to X_data.")
            return
        elif self.Y_data is None :
            print("Please input a data to Y_data.")
            return

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.save()

class DQN(DNN):
    learning_rate=1e-2
    ckpt_name="DQN" + IDENTIFIER
    rewards=None
    prob=None
    low_limit=1
    low_limit_count=0
    
    def on_train_end(self, logs):
        self.global_step+=1

    def __init__(self, state_size, hidden_size=5, output_size=1, layer_level=1, restore=False, ckpt_name=ckpt_name, softmax=True):
        self.ckpt_name = ckpt_name

        self.restore=restore
        self.softmax=softmax

        self._cb_fit.on_train_end=self.on_train_end

        self.state_size=state_size

        if layer_level < 1:
            print("Please set a layer level at least 1.")
            del self
        else:
            self.model = Sequential()

            self.model.add(layers.Input(shape=(state_size,)))
            
            for _ in range(layer_level):
                self.model.add(layers.Dense(hidden_size, kernel_initializer='he_uniform'))
           
            if self.softmax :
                if output_size > 1 :
                    self.model.add(layers.Dense(output_size, activation='softmax'))

                    self.model.compile(optimizer=self.optimizer, loss=self._catcn_loss)
                else :
                    self.model.add(layers.Dense(output_size, activation='sigmoid'))
                    
                    self.model.compile(optimizer=self.optimizer, loss=self._bincn_loss)
            else :
                self.model.add(layers.Dense(output_size))
                
                self.model.compile(optimizer=self.optimizer, loss=self._mse_loss)

            if restore:
                if self.load(self.ckpt_name) is None:
                    print("Create a new model.")

    def _loss_process(self, loss):
        try : return tf.math.pow(1/tf.math.sqrt(tf.reduce_mean(tf.expand_dims(loss,-1) * self.rewards, -1))*10, 2.)
        except : return loss

    def _bincn_loss(self, y_true, y_pred):
        return self._loss_process(losses.binary_crossentropy(y_true, y_pred))

    def _catcn_loss(self, y_true, y_pred):
        return self._loss_process(losses.categorical_crossentropy(y_true, y_pred))

    def _mse_loss(self, y_true, y_pred):
        return self._loss_process(losses.mse(y_true, y_pred))

    def train(self, states, rewards, actions, times=1, reward_std="time"):
        if reward_std=="time":
            self.rewards=self.process_rewards(rewards)
            states=np.array(states)
            rewards=np.array(rewards)
            actions=tf.cast(actions,tf.float32)

            self.low_limit=max(len(rewards),self.low_limit)#(self.low_limit_count/(self.low_limit_count+1))*self.low_limit+(len(rewards)*1.2)/(self.low_limit_count+1)
            #self.low_limit_count+=1

            hist=self.model.fit(states, actions, epochs=times, verbose=0, callbacks=[self._cb_fit], use_multiprocessing=True)
        
        return tf.reduce_mean(hist.history['loss']).numpy()

    def run(self,inputs=None, boolean=True):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)
        
        inputs=np.array(inputs, dtype=np.float32)
        while len(inputs.shape) < 2:
            inputs=np.expand_dims(inputs, -1)

        if self.model is not None and inputs is not None :
            if boolean:
                pred=self.model.predict(inputs, use_multiprocessing=True)
                return np.bool(pred>=0.5)
            else:
                return self.model.predict(inputs, use_multiprocessing=True)
                
    def process_rewards(self, r):
        dr = np.zeros_like(r)

        limit=round(self.low_limit*0.7)
        
        tmp=0
        cnt=0
        for i in range(len(r)-limit,len(r)):
            if i>=0:
                tmp+=r[i]
                cnt+=1
        
        dr[-1]=tmp/cnt*limit

        for i in reversed(range(len(r)-limit,len(r)-1)):
            if i>=0:
                dr[i]=dr[i+1]-r[i+1]
            
        for i in reversed(range(len(r)-limit,len(r))):
            if i>=0:
                dr[i]=1/dr[i]
            
        for i in reversed(range(0,len(r)-limit)):
            if i>=0:
                dr[i]=dr[i+1]+r[i+1]

        #dr[-1] = r[-1]
        #for t in reversed(range(0, len(r)-1)):
        #    dr[t] = dr[t+1] + r[t]
        
        return dr#np.power(dr,2)

class FaceNet:
    BASE_PATH=os.path.dirname(os.path.abspath(__file__))
    threshold=0.35
    registed_face=[]
    face_cascade=None

    def __init__(self, path=None):
        global cv2
        import cv2

        if path is not None:
            self.BASE_PATH=path

        haar_face= '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_face)

        ver=sys.version_info
        self.model_path=self.BASE_PATH+os.path.sep+"model"+os.path.sep+"FaceNet"+os.path.sep+"model"+os.path.sep+str(ver.major)+"."+str(ver.minor)

        if os.path.exists(self.model_path+os.path.sep+"FaceNet.json"):
            try:
                print(1)
                model_file=open(self.model_path+os.path.sep+"FaceNet.json","r")
                model_json=model_file.read()
                model_file.close()
                self.model=model_from_json(model_json)
            except:
                
                self.model=self.create_model()
        else:
            self.model=self.create_model()
        
        self.model.load_weights(self.BASE_PATH+os.path.sep+"model"+os.path.sep+"FaceNet"+os.path.sep+"weights"+os.path.sep+"FaceNet.h5")

    def regist(self, label, data, cropped=True):
        img=deepcopy(data)
        face=img

        if not cropped:
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            det=self.face_cascade.detectMultiScale(gray, scaleFactor=1.3 ,minNeighbors=1,minSize=(100,100))

            for (x1,y1,w,h) in det:
                th=20
            
                x2=x1+w
                y2=y1+h
                
                x1-=th
                y1-=th
                x2+=th*2
                y2+=th*2
                
                if x1<0: x1=0
                if y1<0: y1=0
                if x2>len(img[0]): x2=len(img[0])
                if y2>len(img): y2=len(img)

                face=img[y1:y2,x1:x2]

        face=cv2.resize(face,(160,160))

        pred=self.model.predict(np.array([face]))

        self.registed_face.append({"name":label, "value":pred[0]})

        return face

    def run(self, inputs, cropped=True):
        inputs=tf.keras.preprocessing.sequence.pad_sequences(inputs).astype(np.float32)
        while len(inputs.shape) < 4 :
            inputs=np.expand_dims(inputs, 0)

        preds=self.model.predict(inputs)

        faceid=[]
        for pred in preds:
            dist=100
            i=0
            faceid.append({"name":"Unknown","value":None,"feature":pred})
            for face in self.registed_face:
                x=self._norm(pred)
                y=self._norm(face["value"])
                d=self._EuD(x,y)
                if dist>d :
                    dist=d
                    if self.threshold>d:
                        faceid[i]["name"]=face["name"]
                        faceid[i]["value"]=d
            i+=1
        
        return faceid


    def _norm(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    def _EuD(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def _scaling(self, x, scale):
        return x * scale

    def create_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        inputs = Input(shape=(160, 160, 3))
        x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False, name= 'Conv2d_1a_3x3') (inputs)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_1a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_1a_3x3_Activation')(x)
        x = Conv2D(32, 3, strides=1, padding='valid', use_bias=False, name= 'Conv2d_2a_3x3') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_2a_3x3_Activation')(x)
        x = Conv2D(64, 3, strides=1, padding='same', use_bias=False, name= 'Conv2d_2b_3x3') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2b_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_2b_3x3_Activation')(x)
        x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
        x = Conv2D(80, 1, strides=1, padding='valid', use_bias=False, name= 'Conv2d_3b_1x1') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_3b_1x1_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_3b_1x1_Activation')(x)
        x = Conv2D(192, 3, strides=1, padding='valid', use_bias=False, name= 'Conv2d_4a_3x3') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4a_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_4a_3x3_Activation')(x)
        x = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Conv2d_4b_3x3') (x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4b_3x3_BatchNorm')(x)
        x = Activation('relu', name='Conv2d_4b_3x3_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_1_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_1_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_1_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_1_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_1_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_2_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_2_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_2_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_2_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_2_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_2_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_3_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_3_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_3_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_3_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_3_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_3_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_4_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_4_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_4_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_4_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_4_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_4_Activation')(x)
        
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_5_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_5_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= 'Block35_5_Branch_2_Conv2d_0c_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_5_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= 'Block35_5_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_5_Activation')(x)

        branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_6a_Branch_0_Conv2d_1a_3x3') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_0b_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_1a_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
        branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=3, name='Mixed_6a')(branches)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_1_Branch_1_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_1_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_1_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_1_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_2_Branch_2_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_2_Branch_2_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_2_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_2_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_2_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_3_Branch_3_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_3_Branch_3_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_3_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_3_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_3_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_4_Branch_4_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_4_Branch_4_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_4_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_4_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_4_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_5_Branch_5_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_5_Branch_5_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_5_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_5_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_5_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_6_Branch_6_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_6_Branch_6_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_6_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_6_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_6_Activation')(x)    
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_7_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_7_Branch_7_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_7_Branch_7_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_7_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_7_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_7_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_8_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_8_Branch_8_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_8_Branch_8_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_8_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_8_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_8_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_9_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_9_Branch_9_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_9_Branch_9_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_9_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_9_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_9_Activation')(x)
        
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_10_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0b_1x7') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= 'Block17_10_Branch_10_Conv2d_0c_7x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block17_10_Branch_10_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_10_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= 'Block17_10_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_10_Activation')(x)

        branch_0 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_0a_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')(branch_0)
        branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_1a_3x3') (branch_0)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
        branch_1 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_1a_3x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
        branch_2 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0a_1x1') (x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0b_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_1a_3x3') (branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')(branch_2)
        branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=3, name='Mixed_7a')(branches)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_1_Branch_1_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_1_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_1_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_1_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_2_Branch_2_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_2_Branch_2_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_2_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_2_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_2_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_3_Branch_3_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_3_Branch_3_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_3_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_3_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_3_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_4_Branch_4_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_4_Branch_4_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_4_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_4_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_4_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_5_Branch_5_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_5_Branch_5_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_5_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_5_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_5_Activation')(x)
        
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_0_Conv2d_1x1') (x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0a_1x1') (x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0b_1x3') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= 'Block8_6_Branch_1_Conv2d_0c_3x1') (branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_6_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= 'Block8_6_Conv2d_1x1') (mixed)
        up = Lambda(self._scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 1})(up)
        x = add([x, up])
        
        x = GlobalAveragePooling2D(name='AvgPool')(x)
        x = Dropout(1.0 - 0.8, name='Dropout')(x)
        x = Dense(128, use_bias=False, name='Bottleneck')(x)
        x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)

        model = Model(inputs, x, name='inception_resnet_v1')

        model_json=model.to_json()
        with open(self.model_path+os.path.sep+"FaceNet.json", "w") as json_file : 
            json_file.write(model_json)

        return model



def onehot(array, classes):
    arr=np.array(array)
    return np.squeeze(np.eye(classes)[arr.reshape(-1)])
