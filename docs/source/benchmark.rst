.. _GPU benchmarks:

==============
GPU benchmarks
==============

We have done some benchmarks on our department to find out differences between GPUs and we have
decided to shared them here. Therefore they do not test speed of Neural Monkey, but they 
test different GPU cards with the same setup in Neural Monkey.

The benchmark test consisted of one epoch of Machine Translation training in Neural Monkey
on a set of fixed data. The size of the model nicely fit into the 2GB memory, therefore
GPUs with more memory could have better results with bigger models in comparison to CPUs. 
All GPUs have CUDA8.0

=========================================   ============
Setup (cc=cuda capability)                  Running time 
-----------------------------------------   ------------
GeForce GTX 1080; cc6.1                          9:55:58
GeForce GTX 1080; cc6.1                         10:19:40
GeForce GTX 1080; cc6.1                         12:34:34
GeForce GTX 1080; cc6.1                         13:01:05
GeForce GTX Titan Z; cc3.5                      16:05:24
Tesla K40c; cc3.5                               22:41:01
Tesla K40c; cc3.5                               22:43:10
Tesla K40c; cc3.5                               24:19:45
16 cores Intel Xeon Sandy Bridge 2012 CPU       46:33:14
16 cores Intel Xeon Sandy Bridge 2012 CPU       52:36:56
Quadro K2000; cc3.0                             59:47:58
8 cores Intel Xeon Sandy Bridge 2012 CPU        60:39:17
GeForce GT 630; cc3.0                          103:42:30
8 cores Intel Xeon Westmere 2010 CPU           134:41:22
=========================================   ============
