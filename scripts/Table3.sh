models=(
    'aug01-ce-tor-WRN_40_10',
    'aug01-advbeta2ce-tor-WRN_40_10',
    'aug01-advbetace-tor-WRN_40_10',
    'aug01-advbeta.5ce-tor-WRN_40_10',
    'aug01-advce-tor-WRN_40_10',
    'aug01-strades6ce-tor-WRN_40_10',
    'aug01-strades3ce-tor-WRN_40_10',
    'aug01-stradesce-tor-WRN_40_10',
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

models=(
    'ce-tor-ResNet5-adambs128',
    'advbeta2ce-tor-ResNet5-adambs128',
    'advbetace-tor-ResNet5-adambs128',
    'advbeta.5ce-tor-ResNet5-adambs128',
    'advce-tor-ResNet5-adambs128',
    'strades6ce-tor-ResNet5-adambs128',
    'strades3ce-tor-ResNet5-adambs128',
    'stradesce-tor-ResNet5-adambs128',
)
for i in "${models[@]}"
do
	  python ./main.py --experiment restrictedImgnet \
	    	--no-hooks \
	    	--norm inf --eps 0.005 \
	    	--dataset resImgnet112v3 \
        --model ${i} \
	    	--attack pgd \
	    	--random_seed 0
done
