# dataset
dataset="clothing1m"

# general
seed=$1
gpu=$2

# symmetric
noise_type="none"
noise_ratio=0.4
forget_ratio=0.4
k_ratio=0.7

co_lambda=0.6
lr_ratio=0.001

common="--seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio"
python ./main.py --method_name "standard" $common
python ./main.py --method_name "f-correction" $common
python ./main.py --method_name "decouple" $common
python ./main.py --method_name "co-teaching" --forget_rate $forget_ratio $common
python ./main.py --method_name "co-teaching+" --forget_rate $forget_ratio $common
python ./main.py --method_name "jocor" --forget_rate $forget_ratio --co_lambda $co_lambda $common
python ./main.py --method_name "cdr" $common
python ./main.py --method_name "tv" $common
# python ./main.py --method_name "ours" --k_ratio $k_ratio --lr_ratio $lr_ratio $common