# dataset
datasets=("mnist" "cifar10" "cifar100")

# general
seed=$1
gpu=$2
log_dir="logs/precision"

precisions=(0.2 0.4 0.6 0.8 1.0)

# symmetric
noise_type="symmetric"
noise_ratios=(0.4)

# create dataset
for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        python ./dataset.py --train_ratio 0.9 --seed $seed --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}
    done
done

for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}"
        k_ratio=$(python -c "print(1-${noise_ratios[$j]})")
        for (( l=0; l<${#precisions[@]}; l++ )); do
            python ./main.py --method_name "precision" --k_ratio $k_ratio --precision ${precisions[$l]} --use_multi_k $common
        done
    done
done
