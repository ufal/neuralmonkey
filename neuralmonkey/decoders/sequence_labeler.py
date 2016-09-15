

class SequenceLabeler(Decoder):



    def __init__(self, encoder, vocabulary, data_id, **kwargs):

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id

        # rnn_size property is used in _state_to_output
        self.rnn_size = kwargs.get("hidden_layer_size", 200)

        self.weights, self.biases = self._state_to_output()

        logits = [tf.tanh(tf.matmul(state, self.weights) + self.biases)
                  for state in self.encoder.outputs_bidi]

        self.train_inputs, self.train_weights = self._training_placeholders()
        train_targets = self.train_inputs[1:]

        losses = [tf.nn.softmax_cross_entropy_with_logits(l, t)
                  for l, t in zip(logits, train_targets)]

        weighted_losses = [l * w for l, w in zip(losses, self.train_weights)]

        self.train_loss = sum(weighted_losses)
