# dataset
datasets=("mnist" "cifar10" "cifar100")

# general
seed=$1
gpu=$2
train_ratio=0.8
log_dir="logs/tuning"

lr_ratios=(0.0001 0.0005 0.001 0.005)

# symmetric
noise_type="symmetric"
noise_ratios=(0.2 0.5 0.8)

# create dataset
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./dataset.py --train_ratio 0.9 --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done

k_diffs=(-0.15 -0.1 -0.05 0 0.05 0.1 0.15)
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        for (( l=0; l<${#lr_ratios[@]}; l++ )); do
            for (( m=0; m<${#k_diffs[@]}; m++ )); do
                k_ratio=$(python -c "print(1-${noise_ratios[$k]}+${k_diffs[$m]})")
                python ./main.py --method_name "ours" --k_ratio $k_ratio --lr_ratio ${lr_ratios[$l]} --train_ratio 0.9 --seed $seed --gpu $gpu --log_dir $log_dir --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]} --train_ratio $train_ratio
            done
        done
    done
done

# asymmetric
noise_type="asymmetric"
noise_ratios=(0.4)

# create dataset
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./dataset.py --train_ratio 0.9 --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done

k_diffs=(-0.15 -0.1 -0.05 0 0.05 0.1 0.15)
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        for (( l=0; l<${#lr_ratios[@]}; l++ )); do
            for (( m=0; m<${#k_diffs[@]}; m++ )); do
                k_ratio=$(python -c "print(1-${noise_ratios[$k]}/2+${k_diffs[$m]})")
                python ./main.py --method_name "ours" --k_ratio $k_ratio --lr_ratio ${lr_ratios[$l]} --train_ratio 0.9 --seed $seed --gpu $gpu --log_dir $log_dir --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]} --train_ratio $train_ratio
            done
        done
    done
done