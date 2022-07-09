import tensorflow as tf
import numpy as np

def bert_module(query, key, value, i, modelconfig):
    # Multi headed self-attention
    attention_output, attention_scores = tf.keras.layers.MultiHeadAttention(
        num_heads=modelconfig.NUM_HEAD,
        key_dim=modelconfig.EMBED_DIM // modelconfig.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value, return_attention_scores = True)
    attention_output = tf.keras.layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(modelconfig.FF_DIM, activation="relu"),
            tf.keras.layers.Dense(modelconfig.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = tf.keras.layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output, attention_scores


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc



def create_masked_frame_bert_model(modelconfig):
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_tracker = tf.keras.metrics.Mean()

    class MaskedFrameModel(tf.keras.Model):
        def train_step(self, inputs):
            if len(inputs) == 3:
                features, labels, weights = inputs
            else:
                features, labels = inputs
                weights = None

            with tf.GradientTape() as tape:
                predictions, _ = self(features, training=True)
                loss = loss_fn(labels, predictions, sample_weight = weights)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Compute our own metrics
            loss_tracker.update_state(loss, sample_weight=weights)

            # Return a dict mapping metric names to current value
            return {"loss": loss_tracker.result()}

        def test_step(self, inputs):
            if len(inputs) == 3:
                features, labels, weights = inputs
            else:
                features, labels = inputs
                weights = None

            predictions, _ = self(features, training=False)
            loss = loss_fn(labels, predictions, sample_weight = weights)

            # Compute our own metrics
            loss_tracker.update_state(loss, weights)

            # Return a dict mapping metric names to current value
            return {"val_loss": loss_tracker.result()}

        @property
        def metrics(self):
            # We list our `Metric` objects here so that `reset_states()` can be
            # called automatically at the start of each epoch
            # or at the start of `evaluate()`.
            # If you don't implement this property, you have to call
            # `reset_states()` yourself at the time of your choosing.
            return [loss_tracker]

    inputs = tf.keras.layers.Input((modelconfig.MAX_LEN,modelconfig.INPUT_DIM))
    input_embeddings = tf.keras.layers.Dense(modelconfig.EMBED_DIM, activation='relu')(inputs)

    if modelconfig.USE_POS_EMBED == 1:
        position_embeddings = tf.keras.layers.Embedding(
            input_dim=modelconfig.MAX_LEN,
            output_dim=modelconfig.EMBED_DIM,
            weights=[get_pos_encoding_matrix(modelconfig.MAX_LEN, modelconfig.EMBED_DIM)],
            name="position_embedding",
        )(tf.range(start=0, limit=modelconfig.MAX_LEN, delta=1))
        input_embeddings = input_embeddings + position_embeddings

    encoder_output = input_embeddings
    attention_scores = []
    for i in range(modelconfig.NUM_LAYERS):
        if i == modelconfig.NUM_LAYERS -1:
            model_name = "last"
        else:
            model_name = i
        encoder_output, attention_score = bert_module(encoder_output, encoder_output, encoder_output, model_name, modelconfig)
        attention_scores.append(attention_score)   
    mfm_output = tf.keras.layers.Dense(modelconfig.INPUT_DIM, activation="linear")(
        encoder_output
    )
    mfm_model = MaskedFrameModel(inputs, (mfm_output,attention_scores), name="masked_bert_model")

    return mfm_model
