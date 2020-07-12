models=(
    'ce-tor-WRN_40_10',
    'advbeta2ce-tor-WRN_40_10',
    'advce-tor-WRN_40_10',
    'strades3ce-tor-WRN_40_10',
    'strades6ce-tor-WRN_40_10',

    'ce-tor-WRN_40_10_drop50',
    'advbeta2ce-tor-WRN_40_10_drop50',
    'advce-tor-WRN_40_10_drop50',
    'strades3ce-tor-WRN_40_10_drop50',
    'strades6ce-tor-WRN_40_10_drop50',
)
for i in "${models[@]}"
do
    python ./main.py --experiment experiment01 \
        --no-hooks \
        --norm inf --eps 0.031 \
        --dataset svhn \
        --model ${i} \
        --attack pgd \
        --random_seed 0
done

models=(
    'aug01-ce-tor-WRN_40_10',
    'aug01-advbeta2ce-tor-WRN_40_10',
    'aug01-strades3ce-tor-WRN_40_10',
    'aug01-strades6ce-tor-WRN_40_10',
    'aug01-advce-tor-WRN_40_10-lrem2',

    'aug01-ce-tor-WRN_40_10_drop20',
    'aug01-strades3ce-tor-WRN_40_10_drop20',
    'aug01-strades6ce-tor-WRN_40_10_drop20',
    'aug01-advce-tor-WRN_40_10_drop20-lrem2',
    'aug01-advbeta2ce-tor-WRN_40_10_drop20',
)
for i in "${models[@]}"
do
    python ./main.py --experiment experiment01 \
        --no-hooks \
        --norm inf --eps 0.031 \
        --dataset cifar10 \
        --model ${i} \
        --attack pgd \
        --random_seed 0
done
