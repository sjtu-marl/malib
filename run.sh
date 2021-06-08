name='test'
# GPU=$1
echo "launching container name '${name}'"
sudo docker run --rm -it -v `pwd`:/tmp -t testenv /bin/bash