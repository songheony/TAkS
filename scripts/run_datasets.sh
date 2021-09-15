declare -A k_ratios
declare -A lr_ratios
declare -A co_lambdas

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

# mnist
k_ratios[0,0]=0.65
k_ratios[0,1]=0.35
k_ratios[0,2]=0.15
lr_ratios[0,0]=1
lr_ratios[0,1]=1
lr_ratios[0,2]=1
co_lambdas[0,0]=0.9
co_lambdas[0,1]=0.9
co_lambdas[0,2]=0.9

# cifar10
k_ratios[1,0]=0.75
k_ratios[1,1]=0.45
k_ratios[1,2]=0.2
lr_ratios[1,0]=1
lr_ratios[1,1]=1
lr_ratios[1,2]=1
co_lambdas[1,0]=0.9
co_lambdas[1,1]=0.9
co_lambdas[1,2]=0.9

# cifar100
k_ratios[2,0]=0.65
k_ratios[2,1]=0.35
k_ratios[2,2]=0.15
lr_ratios[2,0]=1
lr_ratios[2,1]=1
lr_ratios[2,2]=1
co_lambdas[2,0]=0.85
co_lambdas[2,1]=0.85
co_lambdas[2,2]=0.85

for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}"
        # python ./main.py --method_name "standard" $common
        # python ./main.py --method_name "f-correction" $common
        # python ./main.py --method_name "decouple" $common
        # python ./main.py --method_name "co-teaching" $common
        # python ./main.py --method_name "co-teaching+" $common
        # python ./main.py --method_name "jocor" --co_lambda ${co_lambdas[$i,$j]} $common
        # python ./main.py --method_name "cdr" $common
        # python ./main.py --method_name "tv" $common
        # python ./main.py --method_name "class2simi" $common --use_pretrained
        # python ./main.py --method_name "taks" --k_ratio ${k_ratios[$i,$j]} --lr_ratio ${lr_ratios[$i,$j]} $common
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

# mnist
k_ratios[0,0]=0.65
k_ratios[0,1]=0.35
k_ratios[0,2]=0.15
lr_ratios[0,0]=1
lr_ratios[0,1]=1
lr_ratios[0,2]=1
co_lambdas[0,0]=0.9
co_lambdas[0,1]=0.9
co_lambdas[0,2]=0.9

# cifar10
k_ratios[1,0]=0.75
k_ratios[1,1]=0.45
k_ratios[1,2]=0.2
lr_ratios[1,0]=1
lr_ratios[1,1]=1
lr_ratios[1,2]=1
co_lambdas[1,0]=0.9
co_lambdas[1,1]=0.9
co_lambdas[1,2]=0.9

# cifar100
k_ratios[2,0]=0.65
k_ratios[2,1]=0.35
k_ratios[2,2]=0.15
lr_ratios[2,0]=1
lr_ratios[2,1]=1
lr_ratios[2,2]=1
co_lambdas[2,0]=0.85
co_lambdas[2,1]=0.85
co_lambdas[2,2]=0.85

for (( i=0; i<${#datasets[@]}; i++ )); do
    for (( j=0; j<${#noise_ratios[@]}; j++ )); do
        common="--train_ratio 0.9 --seed $seed --gpu $gpu --dataset_name ${datasets[$i]} --noise_type $noise_type --noise_ratio ${noise_ratios[$j]}"
        # python ./main.py --method_name "standard" $common
        # python ./main.py --method_name "f-correction" $common
        # python ./main.py --method_name "decouple" $common
        # python ./main.py --method_name "co-teaching" $common
        # python ./main.py --method_name "co-teaching+" $common
        # python ./main.py --method_name "jocor" --co_lambda ${co_lambdas[$i,$j]} $common
        # python ./main.py --method_name "cdr" $common
        # python ./main.py --method_name "tv" $common
        # python ./main.py --method_name "class2simi" $common --use_pretrained
        # python ./main.py --method_name "taks" --k_ratio ${k_ratios[$i,$j]} --lr_ratio ${lr_ratios[$i,$j]} $common
    done
done