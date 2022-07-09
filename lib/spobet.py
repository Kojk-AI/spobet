import tensorflow as tf

def create_model(mfm, modelconfig, dataconfig):
    bert_masked_model = mfm

    bert_backbone = tf.keras.Model(bert_masked_model.input, (bert_masked_model.get_layer("encoder_last/ffn_layernormalization").output[:,0,:], bert_masked_model.output[1]) )
    bert_backbone.trainable=True

    inputs = tf.keras.Input(shape=(modelconfig.MAX_LEN,modelconfig.INPUT_DIM))
    x, attention_scores = bert_backbone(inputs)
    outputs = tf.keras.layers.Dense(dataconfig.NUM_GLOSS, activation='softmax')(x)

    bert_classifier = tf.keras.Model(inputs, outputs)

    return bert_classifier

