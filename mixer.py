import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Mixer(object):

    def __init__(self, decoder):
        self.decoder = decoder

        with tf.variable_scope('mixer'):
            self.bleu = tf.placeholder(tf.float32, [None])

            hidden_states = decoder.hidden_states

            linear_reg_W = tf.Variable(tf.truncated_normal([decoder.rnn_size, 1]))
            linear_reg_b = tf.Variable(tf.zeros([1]))
            
            expected_rewards = [tf.matmul(h, linear_reg_W) + linear_reg_b for h in hidden_states]

            regression_loss = sum([(r - self.bleu) ** 2 for r in expected_rewards]) * 0.5            
            regression_optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(regression_loss)

            ## decoded logits: [batch * slovnik] (delky max sequence) - obsahuje logity
            ## decoded_seq: [batch * 1] (delky max sequence) - obsahuje indexy do slovniku (argmaxy)

            max_logits = [ tf.reduce_max(l, 1) for l in decoder.decoded_logits ] ## batch x 1
            indicator = [tf.to_float(tf.equal(ml, l)) for ml, l in zip(max_logits, decoder.decoded_logits)] ## batch x slovnik

            derivatives = [ tf.reduce_sum((self.bleu - r) *  (tf.nn.softmax(l) - i), 0, keep_dims=True) \
                                for r, l, i in zip(expected_rewards, decoder.decoded_logits, indicator)] ## [1xslovnik](delky max sequence)

            print "DERIVATIVES SHAPE"

            print derivatives[0].get_shape()


            trainable_vars = [v.ref() for v in tf.trainable_variables() if not v.name.startswith('mixer')] 

            print "TRAINABLE VARS"

            for var in trainable_vars:
                print var.get_shape()
                
        

            output_gradients = [tf.gradients(l, trainable_vars, colocate_gradients_with_ops=True)  for l in decoder.decoded_logits] ## [slovnik x shape promenny](delky max seq)

            print "O.G."

            for og in output_gradients[0]:
                print og
            

            reinforce_gradients = [[tf.matmul(d, g) for g in gs] for d, gs in zip(derivatives, output_gradients) ] ## [shape promenny](v case)

            cross_entropies = [tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(l, t) * w, 0) \
                                   for l, t, w in zip(decoder.decoded_logits, decoder.targets, decoder.weights_ins)] ## [skalar](v case)

            xent_gradients = [tf.gradients(e, trainable_vars) for e in cross_entropies]            
            self.mixer_weights = [tf.placeholder(tf.float32, [1]) for _ in hidden_states]

            mixed_gradients = []

            for i, (rgs, xgs, mw) in enumerate(zip(reinforce_gradients, xent_gradients, mixer_weights)):

                for j, (rg, xg) in enumerate(zip(rgs, xgs)):
                    g = xg * mw + rg * (1-mw)

                    if(i == 0):
                        mixed_gradients.append(g)
                    else:
                        mixed_gradients[j] += g

            self.mixer_optimizer = tf.train.AdamOptimizer().apply_gradients(zip(mixed_gradients, trainable_vars))




    def run(self, sess, fd, references):
        decoded_sequence = sess.run(decoder.decoded_seq, fd=fd)
        sentences = decoder.vocabulary.vectors_to_sentences(decoded_sequence)
        bleu_smoothing = SmoothingFunction(epsilon=0.01).method1
        bleus = [sentence_bleu(r, s, smoothing_function=bleu_smoothing) for r, s in zip(references, sentences)]

        fd[self.bleu] = bleus

        for w in self.mixer_weights:
            fd[w] = 1

        sess.run(self.mixed_optimizer, fd=fd)

        
