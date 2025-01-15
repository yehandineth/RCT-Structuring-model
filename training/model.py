import tf_keras as keras
from config.config import *
import tensorflow as tf

def create_model(name='v0.0') -> keras.Model:

    input_character = keras.layers.Input(shape=(CHARACTER_TOKEN_LENGTH, ), dtype=tf.int64, name='chars')

    x1 = keras.layers.Embedding(input_dim=CHARACTER_VOCAB_SIZE,
        output_dim=CHARACTER_EMBEDDING_DIMENSIONS,
        name = 'char_embeddings'
    )(input_character)
    x1 = keras.layers.Bidirectional(keras.layers.LSTM(32), name='BiLSTM_1')(x1)

    input_tokens = keras.layers.Input(shape=(NUM_TOKENS, ), dtype=tf.int64, name='text')
    x2 = keras.layers.Embedding(
        input_dim = VOCAB_SIZE, 
        output_dim =EMBEDDING_DIMENSIONS,
        name = 'text_embeddings'
    )(input_tokens)
    x2 = keras.layers.Conv1D(128,7, activation='relu', name='Conv_1')(x2)
    x2 = keras.layers.MaxPool1D(2, name='Pool_1')(x2)
    x2 = keras.layers.Conv1D(64,3,activation='relu', name='Conv_2')(x2)
    x2 = keras.layers.MaxPool1D(2, name='Pool_2')(x2)
    x2 = keras.layers.GlobalAveragePooling1D(name='G_pool')(x2)
    x2 = keras.layers.Dense(128, activation='relu', name='Text_out')(x2)


    input_lines = keras.layers.Input(shape=(1,), name='line_of')
    x3 = keras.layers.Dense(8, activation='relu', name='Line_of_Dense_1')(input_lines)
    x3 = keras.layers.Dropout(0.5, name='Drop_1')(x3)
    x3 = keras.layers.Dense(8, activation='relu',name='Line_of_Dense_2')(x3)

    x4 = keras.layers.Concatenate(-1, name='concat_1')([x1,x2])
    x4 = keras.layers.Dense(196, activation='relu', name='Dense_2')(x4)
    x4 = keras.layers.Dropout(0.5, name='Drop_2')(x4)

    x = keras.layers.Concatenate(axis=-1, name='concat_2')([x3,x4])

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
            },
        #
        name=name
        )

    model.compile(
            optimizer= keras.optimizers.Adam(name='Adam'), 
            loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.2), 
            metrics=['accuracy']
            )
    return model

def plot_model(model):
    return keras.utils.plot_model(
    model=model, 
    show_dtype=True,
    show_layer_activations=True,
    expand_nested=True,
    show_shapes=True,
    show_trainable=True,
    to_file=SERIALIZATION_DIR.joinpath(f'{model.name}.png')
                       )

def load_model(name=NAME):
    model = create_model(name=name)
    model.optimizer = keras.optimizers.Adam(name='Adam')
    model.load_weights(SERIALIZATION_DIR.joinpath(f'{NAME}.weights.h5'))
    return model