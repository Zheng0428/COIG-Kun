#!/bin/bash

n为迭代次数
n=2 
./ft_seed.sh
./infer_label.sh
for (( i=1; i<=n; i++ ))
do
    echo "loop $i ..."
    ./infer_point.sh
    ./ft_5point.sh
done
./infer_point.sh
echo "finish"

# n=2 
# ./test.sh
# ./test.sh
# for (( i=1; i<=n; i++ ))
# do
#     echo "loop $i ..."
#     ./test.sh
#     ./test.sh
# done
# ./test.sh
# echo "finish"
