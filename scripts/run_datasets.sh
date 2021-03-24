declare -A lr_ratios
declare -A k_ratios
declare -A co_lambdas

# dataset
datasets=("mnist" "cifar10" "cifar100")

# general
seed=$1
gpu=$2

# symmetric
noise_type="symmetric"
noise_ratios=(0.2 0.5 0.8)
forget_ratios=(0.2 0.5 0.8)

# create dataset
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./dataset.py --train_ratio 0.8 --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./dataset.py --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done

# mnist
lr_ratios[0,0]=0.005
lr_ratios[0,1]=0.005
lr_ratios[0,2]=0.005
k_ratios[0,0]=0.65
k_ratios[0,1]=0.35
k_ratios[0,2]=0.15
co_lambdas[0,0]=0.9
co_lambdas[0,1]=0.9
co_lambdas[0,2]=0.9

# cifar10
lr_ratios[1,0]=0.0005
lr_ratios[1,1]=0.001
lr_ratios[1,2]=0.005
k_ratios[1,0]=0.75
k_ratios[1,1]=0.45
k_ratios[1,2]=0.2
co_lambdas[1,0]=0.9
co_lambdas[1,1]=0.9
co_lambdas[1,2]=0.65

# cifar100
lr_ratios[2,0]=0.0005
lr_ratios[2,1]=0.001
lr_ratios[2,2]=0.005
k_ratios[2,0]=0.65
k_ratios[2,1]=0.35
k_ratios[2,2]=0.15
co_lambdas[2,0]=0.85
co_lambdas[2,1]=0.85
co_lambdas[2,2]=0.85

for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./main.py --method_name "standard" --train_ratio 0.8 --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "standard" --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "f-correction" --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "decouple" --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "co-teaching" --forget_rate ${forget_ratios[$k]} --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "co-teaching+" --forget_rate ${forget_ratios[$k]} --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "jocor" --forget_rate ${forget_ratios[$k]} --co_lambda ${co_lambdas[$j,$k]} --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "ours" --k_ratio ${k_ratios[$j,$k]} --lr_ratio ${lr_ratios[$j,$k]} --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done

# asymmetric
noise_type="asymmetric"
noise_ratios=(0.4)
forget_ratios=(0.2)

# create dataset
for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./dataset.py --train_ratio 0.8 --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./dataset.py --seed $seed --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done

# mnist
lr_ratios[0,0]=0.001
k_ratios[0,0]=0.7
co_lambdas[0,0]=0.9

# cifar10
lr_ratios[1,0]=0.0005
k_ratios[1,0]=0.8
co_lambdas[1,0]=0.9

# cifar100
lr_ratios[2,0]=0.0005
k_ratios[2,0]=0.7
co_lambdas[2,0]=0.85

for (( j=0; j<${#datasets[@]}; j++ )); do
    for (( k=0; k<${#noise_ratios[@]}; k++ )); do
        python ./main.py --method_name "standard" --train_ratio 0.8 --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "standard" --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "f-correction" --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "decouple" --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "co-teaching" --forget_rate ${forget_ratios[$k]} --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "co-teaching+" --forget_rate ${forget_ratios[$k]} --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "jocor" --forget_rate ${forget_ratios[$k]} --co_lambda ${co_lambdas[$j,$k]} --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
        python ./main.py --method_name "ours" --k_ratio ${k_ratios[$j,$k]} --lr_ratio ${lr_ratios[$j,$k]} --seed $seed --gpu $gpu --dataset_name ${datasets[$j]} --noise_type $noise_type --noise_ratio ${noise_ratios[$k]}
    done
done