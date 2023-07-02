
from tensorflow import keras

def multivariate_model():
    NUM_FEATURES = 3  # This line is setting the number of features you have in your multivariate data as 3.

    # This line is setting the parameters that will be used to compile your model. The loss function is 'mae' which stands for mean absolute error, and the optimizer is Adam.
    compile_parameters = {'loss' : 'mae' , 'optimizer' : keras.optimizers.legacy.Adam(), 'metrics' : ['mae'] }

    WINDOW_SIZE = 12  # This is the size of the window of data that your model will look at in order to make its next prediction. This means your model will look at the last 12 steps of your time series data to make a prediction for the next step.

    fit_parameters = {'batch_size' : 256, 'epochs' : 130}  # These are parameters for the training process. The batch size is 256, which means 256 samples of data will be passed through the model at each training step. The model will be trained for 130 epochs, which means it will go through the entire training dataset 130 times.

    model = keras.models.Sequential()  # This line creates a new Sequential model. A Sequential model in Keras is a linear stack of layers where you can add one layer at a time.

    # This line adds a Bidirectional LSTM layer to your model. LSTM stands for Long Short-Term Memory and it's a type of recurrent neural network that is good at learning from sequences of data. Bidirectional means that the LSTM layers will learn from the data in both forward and backward time steps. This layer has 64 units.
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64 ,return_sequences=True,input_shape  = (WINDOW_SIZE,NUM_FEATURES)))) 

    model.add(keras.layers.Dropout(0.2))  # This line adds a Dropout layer to your model. Dropout is a regularization technique that randomly ignores some neurons during training, which can help prevent overfitting. The rate 0.2 means that 20% of the neurons will be ignored during training.

    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))  # This line adds another Bidirectional LSTM layer with 64 units. But in this case, return_sequences=False (default value), which means this LSTM layer only returns the last output in the output sequence.

    model.add(keras.layers.Dropout(0.2))  # Another Dropout layer with rate 0.2 is added here.

    model.add(keras.layers.Dense(20,activation = "relu"))  # This line adds a Dense layer (fully connected layer) to your model. This layer has 20 units and uses ReLU (Rectified Linear Unit) as its activation function, which basically means negative values will be zeroed out.

    model.add(keras.layers.Dense(NUM_FEATURES))  # This line adds another Dense layer with the same number of units as you have features in your data. This is usually done in the output layer where you want to predict multiple features.

    model.compile(**compile_parameters)  # This line compiles your model with the parameters specified before. This prepares your model for training.

    return model, fit_parameters, WINDOW_SIZE  # Finally, your function returns the model, the fit parameters, and the window size.


def univariate_model():
    
    compile_parameters = {'loss' : 'mae' , 'optimizer' : keras.optimizers.legacy.Adam(), 'metrics' : ['mse'] }
    fit_parameters = {'batch_size' : 256, 'epochs' : 150}
    WINDOW_SIZE = 36
    model = keras.models.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64 ,return_sequences=True,input_shape  = (WINDOW_SIZE,1)))) 
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(20,activation = "relu"))
    model.add(keras.layers.Dense(1))
    
    model.compile(**compile_parameters)
    
    return model , fit_parameters, WINDOW_SIZE
