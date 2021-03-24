# dataset
datasets=("mnist" "cifar10" "cifar100")

# general
seed=$1
gpu=$2
log_dir="logs/precision"

precisions=(0.2 0.4 0.6 0.8 1.0)

# symmetric
noise_type="symmetric"
noise_ratios=(0.5)

# create dataset
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./dataset.py --train_ratio 0.8 --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./dataset.py --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done

k_ratios=(0.5)
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        for (( l=0; l<${#precisions[@]}; l++ )); do
            python ./main.py --method_name "precision" --k_ratio ${k_ratios[$k]} --precision ${precisions[$l]} --seed $seed --gpu $gpu --log_dir $log_dir --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        done
    done
done
