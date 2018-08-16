# Art-generation-with-neural-style-transfer
Neural Style Transfer (NST) is one of the most fun techniques in deep learning. Basically, it merges two images, namely, a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S. 
I have used following Libraries
1)tensorflow
2)scipy
3)os
4)sys
5)nst_utils
6)PIL
7)matplotlib
8)numpy
Following the original NST paper (https://arxiv.org/abs/1508.06576), we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers)

