import tensorflow as tf

from bayesian_cv.config import ProjectConfig


class MCDropout(tf.keras.layers.Dropout):
    """Dropout layer that stays active during inference for MC sampling."""

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def build_model(config: ProjectConfig) -> tf.keras.Model:
    """
    The model is a simple CNN with dropout layers for MC sampling. 
    You can modify the architecture as needed.
    """
    inputs = tf.keras.Input(
        shape=(*config.image_size, 3),
        name = "input_image")
    
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs) #for å få fine tall

    for filters in config.conv_filters:
        x = tf.keras.layers.Conv2D(filters, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = MCDropout(config.dropout_rate)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(config.dense_units, activation='relu')(x)
    x = MCDropout(config.dropout_rate)(x)
    

    outputs = tf.keras.layers.Dense(config.num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    
    return model


