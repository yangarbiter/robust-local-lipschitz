models=(
    'advbeta2ce-tor-CNN001'
    'advbetace-tor-CNN001'
    'advbeta.5ce-tor-CNN001'
    'strades6ce-tor-CNN001'
    'strades3ce-tor-CNN001'
    'stradesce-tor-CNN001'
    'tulipce-tor-CNN001'
    'ce-tor-CNN001'
    'advce-tor-CNN001'
    'sllrce-tor-CNN001'
)

for i in "${models[@]}"
do
    python ./main.py --experiment experiment01 \
        --no-hooks 
        --norm inf --eps 0.1 \
        --dataset mnist \
        --model ${i} \
        --attack pgd \
        --random_seed 0
done

models=(
    'advbeta2ce-tor-CNN002'
    'advbetace-tor-CNN002'
    'advbeta.5ce-tor-CNN002'
    'strades6ce-tor-CNN002'
    'strades3ce-tor-CNN002'
    'stradesce-tor-CNN002'
    'tulipce-tor-CNN002'
    'ce-tor-CNN002'
    'advce-tor-CNN002'
    'sllrce-tor-CNN002'
)

for i in "${models[@]}"
do
    python ./main.py --experiment experiment01 \
        --no-hooks 
        --norm inf --eps 0.1 \
        --dataset mnist \
        --model ${i} \
        --attack pgd \
        --random_seed 0
done