#!/bin/sh
mkdir -p dataset

url=http://yann.lecun.com/exdb/mnist
wget -P dataset ${url}/train-images-idx3-ubyte.gz
wget -P dataset ${url}/train-labels-idx1-ubyte.gz
wget -P dataset ${url}/t10k-images-idx3-ubyte.gz
wget -P dataset ${url}/t10k-labels-idx1-ubyte.gz

gzip -d dataset/*.gz