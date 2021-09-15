# dataset
datasets=("mnist" "cifar10" "cifar100")

# general
seed=$1
gpu=$2

# symmetric
noise_type="symmetric"
noise_ratios=(0.2 0.4 0.6)

# create dataset
for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        python ./dataset.py --train_ratio 0.9 --seed $seed --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}
    done
done

# tune hyper-parameters
k_diffs=(-0.15 -0.1 -0.05 0 0.05 0.1 0.15)
lr_ratios=(1 0.1 0.001 0.0001)
for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}"
        for (( l=0; l<${#lr_ratios[@]}; l++ )); do
            # for (( k=0; k<${#k_diffs[@]}; k++ )); do
            #     k_ratio=$(python -c "print(1-${noise_ratios[$j]}+${k_diffs[$k]})")
            #     python ./main.py --method_name "taks" --use_total --use_noise --k_ratio $k_ratio --lr_ratio ${lr_ratios[$l]} $common
            #     python ./main.py --method_name "taks" --use_total --use_noise --k_ratio $k_ratio --lr_ratio ${lr_ratios[$l]} --use_multi_k $common
            # done
            python ./main.py --method_name "taks" --use_total --use_noise --k_ratio 0 --lr_ratio ${lr_ratios[$l]} $common
        done
    done
done

# asymmetric
noise_type="asymmetric"
noise_ratios=(0.2 0.4 0.6)

# create dataset
for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        python ./dataset.py --train_ratio 0.9 --seed $seed --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}
    done
done

# tune hyper-parameters
k_diffs=(-0.15 -0.1 -0.05 0 0.05 0.1 0.15)
for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}"
        for (( l=0; l<${#lr_ratios[@]}; l++ )); do
            # for (( k=0; k<${#k_diffs[@]}; k++ )); do
            #     k_ratio=$(python -c "print(1-${noise_ratios[$j]}+${k_diffs[$k]})")
            #     python ./main.py --method_name "taks" --use_total --use_noise --k_ratio $k_ratio --lr_ratio ${lr_ratios[$l]} $common
            #     python ./main.py --method_name "taks" --use_total --use_noise --k_ratio $k_ratio --lr_ratio ${lr_ratios[$l]} --use_multi_k True $common
            # done
            python ./main.py --method_name "taks" --use_total --use_noise --k_ratio 0 --lr_ratio ${lr_ratios[$l]} $common
        done
    done
done
