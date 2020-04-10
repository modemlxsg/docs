import tensorflow as tf



def decode(sequence):
    inputs = tf.constant(sequence)
    sequence_length = tf.constant([inputs.shape[0]] * inputs.shape[1])
    decoded, _ = tf.nn.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True)
    # decoded, _ = tf.nn.ctc_beam_search_decoder(inputs, sequence_length)
    decoded = tf.sparse.to_dense(decoded[0]).numpy()
    return decoded
