import tf_keras as keras
from config.config import *
import tensorflow as tf

def create_model() -> keras.Model:

    input_character = keras.layers.Input(shape=(CHARACTER_TOKEN_LENGTH, ), dtype=tf.int64, name='chars')

    x1 = keras.layers.Embedding(input_dim=CHARACTER_VOCAB_SIZE,
        output_dim=CHARACTER_EMBEDDING_DIMENSIONS,
        name = 'char_embeddings'
    )(input_character)
    x1 = keras.layers.Bidirectional(keras.layers.LSTM(32))(x1)

    input_tokens = keras.layers.Input(shape=(NUM_TOKENS, ), dtype=tf.int64, name='text')
    x2 = keras.layers.Embedding(
        input_dim = VOCAB_SIZE, 
        output_dim =EMBEDDING_DIMENSIONS,
        name = 'text_embeddings'
    )(input_tokens)
    x2 = keras.layers.Conv1D(128,7, activation='relu')(x2)
    x2 = keras.layers.MaxPool1D(2)(x2)
    x2 = keras.layers.Conv1D(64,3,activation='relu')(x2)
    x2 = keras.layers.MaxPool1D(2)(x2)
    x2 = keras.layers.GlobalAveragePooling1D()(x2)
    x2 = keras.layers.Dense(128, activation='relu')(x2)


    input_lines = keras.layers.Input(shape=(1,), name='line_of')
    x3 = keras.layers.Dense(8, activation='relu')(input_lines)
    x3 = keras.layers.Dropout(0.5)(x3)
    x3 = keras.layers.Dense(8, activation='relu')(x3)

    x4 = keras.layers.Concatenate(-1)([x1,x2])
    x4 = keras.layers.Dense(196, activation='relu')(x4)
    x4 = keras.layers.Dropout(0.5)(x4)

    x = keras.layers.Concatenate(axis=-1)([x3,x4])

    x = keras.layers.Dense(NUM_CLASSES, name='classifier')(x)
    outputs = keras.layers.Activation(keras.activations.softmax, dtype=tf.float32, name='outputs')(x)

    model = keras.Model(
        inputs = {
                'text' : input_tokens,
                'chars' : input_character,
                'line_of' : input_lines,
            },
        outputs= {
            'outputs' : outputs
            }
        )

    model.compile(
            optimizer= keras.optimizers.Adam(), 
            loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.2), 
            metrics=['accuracy']
            )
    plot_model(model)
    return model

def plot_model(model):
    return keras.utils.plot_model(
    model=model, 
    show_dtype=True,
    show_layer_activations=True,
    expand_nested=True,
    show_shapes=True,
    show_trainable=True
                       )