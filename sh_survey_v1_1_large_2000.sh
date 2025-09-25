#!/bin/bash

# ##############################zero shot##############################
# # # # 设置模型列表
# MODELS=("Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B")   # "Qwen2.5-7B"   "DeepSeek-R1-Distill-Qwen-7B""llama2-70b"   "Llama-3.1-8B-Instruct" "Qwen2.5-Math-7B-Instruct" "Qwen2.5-Math-7B"  "DeepSeek-R1-Distill-Qwen-7B"

# TYPE="completion"
# LOCATION="sf"
# TRAVEL_FILE="travel"
# LOC_UB="none"
# USE_POPN_SAMPLING="true"
# USE_DATE_SAMPLING="true"
# FIX_BIAS="false"
# YEAR="2017"
# EPOCH=1
# PYTHON_SCRIPT="main1_1.py"
# FEW_SHOT="false"
# PROFILE_TYPE="actual"

# for MODEL in "${MODELS[@]}"; do
#     OUTPUT_FOLDER="outputs/v11/zero_shot/${MODEL}"

#     echo "Running experiment for model: $MODEL"
#     mkdir -p "$OUTPUT_FOLDER"

#     CUDA_VISIBLE_DEVICES=2,3 python "$PYTHON_SCRIPT" \
#         --type "$TYPE" \
#         --location "$LOCATION" \
#         --travel_file "$TRAVEL_FILE" \
#         --loc_ub "$LOC_UB" \
#         --use_popn_sampling "$USE_POPN_SAMPLING" \
#         --use_date_sampling "$USE_DATE_SAMPLING" \
#         --use_model "$MODEL" \
#         --out_folder "$OUTPUT_FOLDER" \
#         --fix_bias "$FIX_BIAS" \
#         --year "$YEAR" \
#         --trained_model_epoch "$EPOCH" \
#         --few_shot "$FEW_SHOT" \
#         --profile_type "$PROFILE_TYPE" 
# done


# # #######################################few shot##############################
# MODELS=("Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B")    #  "Qwen2.5-Math-7B""DeepSeek-R1-Distill-Qwen-7B"
# TYPE="completion"
# LOCATION="sf"
# TRAVEL_FILE="travel"
# LOC_UB="none"
# USE_POPN_SAMPLING="true"
# USE_DATE_SAMPLING="true"
# FIX_BIAS="false"
# YEAR="2017"
# EPOCH=1
# PYTHON_SCRIPT="main1_1.py"
# FEW_SHOT="true"
# PROFILE_TYPE="actual"

# for MODEL in "${MODELS[@]}"; do
#     OUTPUT_FOLDER="outputs/v11/few_shot/${MODEL}"

#     echo "Running experiment for model: $MODEL"
#     mkdir -p "$OUTPUT_FOLDER"

#     CUDA_VISIBLE_DEVICES=2,3 python "$PYTHON_SCRIPT" \
#         --type "$TYPE" \
#         --location "$LOCATION" \
#         --travel_file "$TRAVEL_FILE" \
#         --loc_ub "$LOC_UB" \
#         --use_popn_sampling "$USE_POPN_SAMPLING" \
#         --use_date_sampling "$USE_DATE_SAMPLING" \
#         --use_model "$MODEL" \
#         --out_folder "$OUTPUT_FOLDER" \
#         --fix_bias "$FIX_BIAS" \
#         --year "$YEAR" \
#         --trained_model_epoch "$EPOCH" \
#         --few_shot "$FEW_SHOT" \
#         --profile_type "$PROFILE_TYPE"
# done

##################################RL zero shot##############################
# 设置模型列表    "Llama-3.1-8B-GRPO-lm2-v1" 
# MODELS=(DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v3-500)   #"Llama-3.1-8B-GRPO6-merged" "llama2-70b"   "Llama-3.1-8B-Instruct" "Qwen2.5-Math-7B-Instruct" "Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-trained" 
# # "Llama-3.1-8B-trained_baseline" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-GRPO8-lm2""DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v2"
# TYPE="completion"
# LOCATION="sf"
# TRAVEL_FILE="travel"
# LOC_UB="none"
# USE_POPN_SAMPLING="true"
# USE_DATE_SAMPLING="true"
# FIX_BIAS="false"
# YEAR="2017"
# EPOCH=1
# PYTHON_SCRIPT="main1_1_large_2000.py"
# FEW_SHOT="false"
# PROFILE_TYPE="actual"
# LOCATION_TEST=("SF" "ATX" "ATL" "SD")  # "ATL" "SF"

# for location_test in "${LOCATION_TEST[@]}"; do
#     echo "Running experiments for location: $location_test"
#     for MODEL in "${MODELS[@]}"; do
#         OUTPUT_FOLDER="outputs/sft/v11_large_2000/zero_shot/${location_test}/${MODEL}"
#         # OUTPUT_FOLDER="outputs/sft/v2/${PROFILE_TYPE}/zero_shot/${MODEL}"   
#         echo "Running experiment for model: $MODEL"
#         mkdir -p "$OUTPUT_FOLDER"

#         python "$PYTHON_SCRIPT" \
#             --type "$TYPE" \
#             --location "$LOCATION" \
#             --travel_file "$TRAVEL_FILE" \
#             --loc_ub "$LOC_UB" \
#             --use_popn_sampling "$USE_POPN_SAMPLING" \
#             --use_date_sampling "$USE_DATE_SAMPLING" \
#             --use_model "$MODEL" \
#             --out_folder "$OUTPUT_FOLDER" \
#             --fix_bias "$FIX_BIAS" \
#             --year "$YEAR" \
#             --trained_model_epoch "$EPOCH" \
#             --few_shot "$FEW_SHOT" \
#             --profile_type "$PROFILE_TYPE" \
#             --location_test "$location_test"
#     done
# done

#################################RL zero shot new##############################
# 设置模型列表    "Llama-3.1-8B-GRPO-lm2-v1" 
MODELS=("DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v3-500" "DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v2")   #"Llama-3.1-8B-GRPO6-merged" "llama2-70b"   "Llama-3.1-8B-Instruct" "Qwen2.5-Math-7B-Instruct" "Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-trained" 
# "Llama-3.1-8B-trained_baseline" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-GRPO8-lm2""DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v2"
TYPE="completion"
LOCATION="sf"
TRAVEL_FILE="travel"
LOC_UB="none"
USE_POPN_SAMPLING="true"
USE_DATE_SAMPLING="true"
FIX_BIAS="false"
YEAR="2017"
EPOCH=1
PYTHON_SCRIPT="main1_1_large_2000.py"
FEW_SHOT="false"
PROFILE_TYPE="actual"
LOCATION_TEST=("SF")  # "ATL" "SF" "ATX" "ATL" "SD"

for location_test in "${LOCATION_TEST[@]}"; do
    echo "Running experiments for location: $location_test"
    for MODEL in "${MODELS[@]}"; do
        OUTPUT_FOLDER="outputs/sft/v11_large_2000/zero_shot/${location_test}/${MODEL}"
        # OUTPUT_FOLDER="outputs/sft/v2/${PROFILE_TYPE}/zero_shot/${MODEL}"   
        echo "Running experiment for model: $MODEL"
        mkdir -p "$OUTPUT_FOLDER"

        python "$PYTHON_SCRIPT" \
            --type "$TYPE" \
            --location "$LOCATION" \
            --travel_file "$TRAVEL_FILE" \
            --loc_ub "$LOC_UB" \
            --use_popn_sampling "$USE_POPN_SAMPLING" \
            --use_date_sampling "$USE_DATE_SAMPLING" \
            --use_model "$MODEL" \
            --out_folder "$OUTPUT_FOLDER" \
            --fix_bias "$FIX_BIAS" \
            --year "$YEAR" \
            --trained_model_epoch "$EPOCH" \
            --few_shot "$FEW_SHOT" \
            --profile_type "$PROFILE_TYPE" \
            --location_test "$location_test"
    done
done


#################################sft zero shot##############################   enhance
# 设置模型列表
MODELS=("DeepSeek-R1-Distill-Llama-8B-trained_enhance" )   # "llama2-70b"   "Llama-3.1-8B-Instruct" "Qwen2.5-Math-7B-Instruct" "Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-trained" 
# "Llama-3.1-8B-trained_baseline" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-trained"
TYPE="completion"
LOCATION="sf"
TRAVEL_FILE="travel"
LOC_UB="none"
USE_POPN_SAMPLING="true"
USE_DATE_SAMPLING="true"
FIX_BIAS="false"
YEAR="2017"
EPOCH=1
PYTHON_SCRIPT="main1_1_large_2000.py"
FEW_SHOT="false"
PROFILE_TYPE="actual"
LOCATION_TEST=("SF" "ATX" "ATL" "SD") 

for location_test in "${LOCATION_TEST[@]}"; do
    echo "Running experiments for location: $location_test"
    for MODEL in "${MODELS[@]}"; do
        # OUTPUT_FOLDER="outputs/sft/v11_large_2000/zero_shot/${MODEL}"
        OUTPUT_FOLDER="outputs/sft/v11_large_2000/zero_shot/${location_test}/${MODEL}"
        # OUTPUT_FOLDER="outputs/sft/v2/${PROFILE_TYPE}/zero_shot/${MODEL}"   
        echo "Running experiment for model: $MODEL"
        mkdir -p "$OUTPUT_FOLDER"

        python "$PYTHON_SCRIPT" \
            --type "$TYPE" \
            --location "$LOCATION" \
            --travel_file "$TRAVEL_FILE" \
            --loc_ub "$LOC_UB" \
            --use_popn_sampling "$USE_POPN_SAMPLING" \
            --use_date_sampling "$USE_DATE_SAMPLING" \
            --use_model "$MODEL" \
            --out_folder "$OUTPUT_FOLDER" \
            --fix_bias "$FIX_BIAS" \
            --year "$YEAR" \
            --trained_model_epoch "$EPOCH" \
            --few_shot "$FEW_SHOT" \
            --profile_type "$PROFILE_TYPE" \
            --location_test "$location_test"
    done
done    

# LLaMA-Factory/model/lora/v11/Llama-3.1-8B-GRPO5-merged

# sh TRL_GRPO/vllm_server.sh

##################################RL zero shot##############################
# 设置模型列表    "Llama-3.1-8B-GRPO-lm2-v1" 
# MODELS=(DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v3-500)   #"Llama-3.1-8B-GRPO6-merged" "llama2-70b"   "Llama-3.1-8B-Instruct" "Qwen2.5-Math-7B-Instruct" "Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-trained" 
# # "Llama-3.1-8B-trained_baseline" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-GRPO8-lm2""DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v2"
# TYPE="completion"
# LOCATION="sf"
# TRAVEL_FILE="travel"
# LOC_UB="none"
# USE_POPN_SAMPLING="true"
# USE_DATE_SAMPLING="true"
# FIX_BIAS="false"
# YEAR="2017"
# EPOCH=1
# PYTHON_SCRIPT="main1_1_large_2000.py"
# FEW_SHOT="false"
# PROFILE_TYPE="actual"
# LOCATION_TEST=("SF" "ATX" "ATL" "SD")  # "ATL" "SF"

# for location_test in "${LOCATION_TEST[@]}"; do
#     echo "Running experiments for location: $location_test"
#     for MODEL in "${MODELS[@]}"; do
#         OUTPUT_FOLDER="outputs/sft/v11_large_2000/zero_shot/${location_test}/${MODEL}"
#         # OUTPUT_FOLDER="outputs/sft/v2/${PROFILE_TYPE}/zero_shot/${MODEL}"   
#         echo "Running experiment for model: $MODEL"
#         mkdir -p "$OUTPUT_FOLDER"

#         python "$PYTHON_SCRIPT" \
#             --type "$TYPE" \
#             --location "$LOCATION" \
#             --travel_file "$TRAVEL_FILE" \
#             --loc_ub "$LOC_UB" \
#             --use_popn_sampling "$USE_POPN_SAMPLING" \
#             --use_date_sampling "$USE_DATE_SAMPLING" \
#             --use_model "$MODEL" \
#             --out_folder "$OUTPUT_FOLDER" \
#             --fix_bias "$FIX_BIAS" \
#             --year "$YEAR" \
#             --trained_model_epoch "$EPOCH" \
#             --few_shot "$FEW_SHOT" \
#             --profile_type "$PROFILE_TYPE" \
#             --location_test "$location_test"
#     done
# done

# sh TRL_GRPO/vllm_server.sh

###################################baseline zero shot##############################
# # 设置模型列表
# MODELS=("DeepSeek-V3")   #"Llama-3.1-8B-GRPO6-merged" "llama2-70b"   "Llama-3.1-8B-Instruct" "Qwen2.5-Math-7B-Instruct" "Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-trained" 
# # "Llama-3.1-8B-trained_baseline" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-GRPO8-lm2"
# TYPE="completion"
# LOCATION="sf"
# TRAVEL_FILE="travel"
# LOC_UB="none"
# USE_POPN_SAMPLING="true"
# USE_DATE_SAMPLING="true"
# FIX_BIAS="false"
# YEAR="2017"
# EPOCH=1
# PYTHON_SCRIPT="main1_1.py"
# FEW_SHOT="false"
# PROFILE_TYPE="actual"

# for MODEL in "${MODELS[@]}"; do
#     OUTPUT_FOLDER="outputs/v111/zero_shot/${MODEL}"
#     # OUTPUT_FOLDER="outputs/sft/v2/${PROFILE_TYPE}/zero_shot/${MODEL}"   
#     echo "Running experiment for model: $MODEL"
#     mkdir -p "$OUTPUT_FOLDER"

#     python "$PYTHON_SCRIPT" \
#         --type "$TYPE" \
#         --location "$LOCATION" \
#         --travel_file "$TRAVEL_FILE" \
#         --loc_ub "$LOC_UB" \
#         --use_popn_sampling "$USE_POPN_SAMPLING" \
#         --use_date_sampling "$USE_DATE_SAMPLING" \
#         --use_model "$MODEL" \
#         --out_folder "$OUTPUT_FOLDER" \
#         --fix_bias "$FIX_BIAS" \
#         --year "$YEAR" \
#         --trained_model_epoch "$EPOCH" \
#         --few_shot "$FEW_SHOT" \
#         --profile_type "$PROFILE_TYPE"
# done

#############################COPB LLMoB############################
MODELS=("DeepSeek-R1-Distill-Llama-8B")   #"Llama-3.1-8B-GRPO6-merged" "llama2-70b"   "Llama-3.1-8B-Instruct" "Qwen2.5-Math-7B-Instruct" "Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-trained" 
# "Llama-3.1-8B-trained_baseline" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-GRPO8-lm2""DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v2"
TYPE="completion"
LOCATION="sf"
TRAVEL_FILE="travel"
LOC_UB="none"
USE_POPN_SAMPLING="true"
USE_DATE_SAMPLING="true"
FIX_BIAS="false"
YEAR="2017"
EPOCH=1
PYTHON_SCRIPT="main1_1_large_2000.py"
FEW_SHOT="LLMob"
PROFILE_TYPE="actual"
LOCATION_TEST=("SF" "ATX" "ATL" "SD")  # "ATL" "SF"

for location_test in "${LOCATION_TEST[@]}"; do
    echo "Running experiments for location: $location_test"
    for MODEL in "${MODELS[@]}"; do
        OUTPUT_FOLDER="outputs/sft/v11_large_2000/zero_shot/${FEW_SHOT}/${location_test}/${MODEL}"
        # OUTPUT_FOLDER="outputs/sft/v2/${PROFILE_TYPE}/zero_shot/${MODEL}"   
        echo "Running experiment for model: $MODEL"
        mkdir -p "$OUTPUT_FOLDER"

        python "$PYTHON_SCRIPT" \
            --type "$TYPE" \
            --location "$LOCATION" \
            --travel_file "$TRAVEL_FILE" \
            --loc_ub "$LOC_UB" \
            --use_popn_sampling "$USE_POPN_SAMPLING" \
            --use_date_sampling "$USE_DATE_SAMPLING" \
            --use_model "$MODEL" \
            --out_folder "$OUTPUT_FOLDER" \
            --fix_bias "$FIX_BIAS" \
            --year "$YEAR" \
            --trained_model_epoch "$EPOCH" \
            --few_shot "$FEW_SHOT" \
            --profile_type "$PROFILE_TYPE" \
            --location_test "$location_test"
    done
done



#############################COPB LLMoB############################
MODELS=("DeepSeek-R1-Distill-Llama-8B")   #"Llama-3.1-8B-GRPO6-merged" "llama2-70b"   "Llama-3.1-8B-Instruct" "Qwen2.5-Math-7B-Instruct" "Llama-3.1-8B" "DeepSeek-R1-Distill-Llama-8B" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-trained" 
# "Llama-3.1-8B-trained_baseline" "Llama-3.1-8B-trained" "DeepSeek-R1-Distill-Llama-8B-GRPO8-lm2""DeepSeek-R1-Distill-Llama-8B-GRPO-lm2-v2"
TYPE="completion"
LOCATION="sf"
TRAVEL_FILE="travel"
LOC_UB="none"
USE_POPN_SAMPLING="true"
USE_DATE_SAMPLING="true"
FIX_BIAS="false"
YEAR="2017"
EPOCH=1
PYTHON_SCRIPT="main1_1_large_2000.py"
FEW_SHOT="CoPB"
PROFILE_TYPE="actual"
LOCATION_TEST=("SF" "ATX" "ATL" "SD")  # "ATL" "SF"

for location_test in "${LOCATION_TEST[@]}"; do
    echo "Running experiments for location: $location_test"
    for MODEL in "${MODELS[@]}"; do
        OUTPUT_FOLDER="outputs/sft/v11_large_2000/zero_shot/${FEW_SHOT}/${location_test}/${MODEL}"
        # OUTPUT_FOLDER="outputs/sft/v2/${PROFILE_TYPE}/zero_shot/${MODEL}"   
        echo "Running experiment for model: $MODEL"
        mkdir -p "$OUTPUT_FOLDER"

        python "$PYTHON_SCRIPT" \
            --type "$TYPE" \
            --location "$LOCATION" \
            --travel_file "$TRAVEL_FILE" \
            --loc_ub "$LOC_UB" \
            --use_popn_sampling "$USE_POPN_SAMPLING" \
            --use_date_sampling "$USE_DATE_SAMPLING" \
            --use_model "$MODEL" \
            --out_folder "$OUTPUT_FOLDER" \
            --fix_bias "$FIX_BIAS" \
            --year "$YEAR" \
            --trained_model_epoch "$EPOCH" \
            --few_shot "$FEW_SHOT" \
            --profile_type "$PROFILE_TYPE" \
            --location_test "$location_test"
    done
done


sh TRL_GRPO/vllm_server.sh