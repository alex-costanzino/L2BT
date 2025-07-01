export CUDA_VISIBLE_DEVICES=0

class_names=("candle" "capsules" "cashew" "chewinggum" "fryum" "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum")

for class_name in "${class_names[@]}"
    do
        python inference_visa_auto.py --class_name $class_name 
    done