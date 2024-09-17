# model.py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ["OMP_NUM_THREADS"] = '4'
print("OMP_NUM_THREADS =", os.environ['OMP_NUM_THREADS'])
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def create_model(nn_params, input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    hl = layers.Dense(50, activation=nn_params['act_func'],
                      kernel_regularizer=nn_params['l2_reg'])(input_layer)
    for hl_idx in range(1, nn_params['nhidden_layer']):
        hl = layers.Dense(20, activation=nn_params['act_func'],
                          kernel_regularizer=nn_params['l2_reg'])(hl)
        if (hl_idx % 2 == 0) and nn_params['dropout']:
            hl = layers.Dropout(0.3)(hl)
        if (hl_idx % 2 == 1) and nn_params['bnorm']:
            hl = layers.BatchNormalization()(hl)
    output_layer = layers.Dense(1, activation='linear',
                                kernel_regularizer=nn_params['l2_reg'])(hl)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mean_squared_error')
    return model

def train_model(model, train_X, train_y, val_X, val_y, epochs=300):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=100,
        verbose=1,
        restore_best_weights=True
    )
    history = model.fit(train_X, train_y,
                        epochs=epochs,
                        validation_data=(val_X, val_y),
                        callbacks=[early_stopping],
                        verbose=1)
    model.save('model/my_model.keras')  # Save as Keras format
    return history

def load_model():
    return tf.keras.models.load_model('model/my_model.keras')  # Load Keras format
