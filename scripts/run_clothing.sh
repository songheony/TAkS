# dataset
dataset="clothing1m"

# general
seed=$1
gpu=$2

# symmetric
noise_type="symmetric"
noise_ratio=0.4
forget_ratio=0.4
k_ratio=0.7

co_lambda=0.6
lr_ratio=0.001

python ./main.py --method_name "standard" --seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio
python ./main.py --method_name "f-correction" --seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio
python ./main.py --method_name "decouple" --seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio
python ./main.py --method_name "co-teaching" --forget_rate $forget_ratio --seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio
python ./main.py --method_name "co-teaching+" --forget_rate $forget_ratio --seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio
python ./main.py --method_name "jocor" --forget_rate $forget_ratio --co_lambda $co_lambda --seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio
python ./main.py --method_name "ours" --k_ratio $k_ratio --lr_ratio $lr_ratio --seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio
python ./main.py --method_name "itlm" --forget_rate $forget_ratio --seed $seed --gpu $gpu --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio