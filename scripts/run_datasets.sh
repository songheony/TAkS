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

for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}"
        # python ./main.py --method_name "standard" $common
        # python ./main.py --method_name "f-correction" $common
        # python ./main.py --method_name "decouple" $common
        # python ./main.py --method_name "co-teaching" $common
        # python ./main.py --method_name "co-teaching+" $common
        # python ./main.py --method_name "jocor" $common
        # python ./main.py --method_name "cdr" $common
        # python ./main.py --method_name "tv" $common
        # python ./main.py --method_name "class2simi" $common --use_pretrained
        # python ./main.py --method_name "taks" $common --use_pretrained
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

for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}"
        # python ./main.py --method_name "standard" $common
        # python ./main.py --method_name "f-correction" $common
        # python ./main.py --method_name "decouple" $common
        # python ./main.py --method_name "co-teaching" $common
        # python ./main.py --method_name "co-teaching+" $common
        # python ./main.py --method_name "jocor" $common
        # python ./main.py --method_name "cdr" $common
        # python ./main.py --method_name "tv" $common
        # python ./main.py --method_name "class2simi" $common --use_pretrained
        # python ./main.py --method_name "taks" $common --use_pretrained
    done
done