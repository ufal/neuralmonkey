#Trainers

Classes that run the training process.

- `cross_entropy_trainer.py` trains the network by minimizing the cross entropy. This is pretty much the default.
- `copy_net_trainer.py` implements training with Copynet (http://arxiv.org/abs/1603.06393)
- `mixer.py` implements reinforcement learning with MIXER (http://arxiv.org/abs/1511.06732)