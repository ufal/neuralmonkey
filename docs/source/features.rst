.. _features:

=================
Advanced Features
=================

Byte Pair Encoding
------------------

This is explained in
:ref:`the machine translation tutorial <machine-translation>`.

Dropout
-------

Neural networks with a large number of parameters have a serious problem with an overfitting. 
Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural
network during training. This prevents units from co-adapting too much. But during the test time, the dropout is turned off. More information in https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

If you want to enable dropout on an encoder or on the decoder, you can simply add dropout_keep_prob to the particular section::
  
  [encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  dropout_keep_prob=0.8
  ...

or::
 
  [decoder]
  class=decoders.decoder.Decoder
  dropout_keep_prob=0.8
  ...

Pervasive Dropout
-----------------

Detailed information in https://arxiv.org/abs/1512.05287

If you want allow dropout on the recurrent layer of your encoder, you can add use_pervasive_dropout parameter into it and then the dropout probability will be used::

  [encoder]
  class=encoders.sentence_encoder.SentenceEncoder
  dropout_keep_prob=0.8
  use_pervasive_dropout=True
  ...

Attention Seeded by GIZA++ Word Alignments
------------------------------------------

todo: OC to reference the paper and describe how to use this in NM
