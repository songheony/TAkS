# dataset
datasets=("mnist" "cifar10" "cifar100")

# general
seed=$1
gpu=$2

# symmetric
noise_type="symmetric"
noise_ratios=(0.2 0.4 0.6)

# create dataset
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./dataset.py --train_ratio 0.9 --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done

# tune hyper-parameters
k_diffs=(-0.15 -0.1 -0.05 0 0.05 0.1 0.15)
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}"
        for (( m=0; m<${#k_diffs[@]}; m++ )); do
            k_ratio=$(python -c "print(1-${noise_ratios[$k]}+${k_diffs[$m]})")
            python ./main.py --method_name "taks" --k_ratio $k_ratio $common
        done
    done
done

# asymmetric
noise_type="asymmetric"
noise_ratios=(0.2 0.4 0.6)

# create dataset
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./dataset.py --train_ratio 0.9 --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done

# tune hyper-parameters
k_diffs=(-0.15 -0.1 -0.05 0 0.05 0.1 0.15)
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}"
        for (( m=0; m<${#k_diffs[@]}; m++ )); do
            k_ratio=$(python -c "print(1-${noise_ratios[$k]}+${k_diffs[$m]})")
            python ./main.py --method_name "taks" --k_ratio $k_ratio $common
        done
    done
done
