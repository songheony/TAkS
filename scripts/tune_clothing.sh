# dataset
dataset="clothing1m"

# general
seed=$1
gpu=$2
log_dir="logs/tuning"

lr_ratios=(0.0001 0.0005 0.001 0.005)

# symmetric
noise_type="symmetric"
noise_ratio=0.4
k_diffs=(-0.1 -0.05 0 0.05 0.1)

for (( l=0; l<${#lr_ratios[@]}; l++ )); do
    for (( m=0; m<${#k_diffs[@]}; m++ )); do
        k_ratio=$(python -c "print(1-$noise_ratio+${k_diffs[$m]})")
        python ./main.py --method_name "ours" --k_ratio $k_ratio --lr_ratio ${lr_ratios[$l]} --seed $seed --gpu $gpu --log_dir $log_dir --dataset_name $dataset --noise_type $noise_type --noise_ratio $noise_ratio
    done
done