import tensorflow as tf


def get_ann():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(29,)),)
    model.add(tf.keras.layers.Dense(256, activation='relu', 
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                    ),)
                                    
    model.add(tf.keras.layers.Dense(256, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                    ),)
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'),)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model