#!/bin/bash

##########################main-result###########################

LOCATION_TEST=("SF" "ATX" "ATL" "SD")  # "ATL" "SF"

for location_test in "${LOCATION_TEST[@]}"; do

    ########################zero shot###########################
    echo "Processing completion outputs llama(llama zero shot)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/${location_test}/Llama-3.1-8B/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_llama --fix_missing False

    ########################few shot###########################
    echo "Processing completion outputs llama(llama few shot)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/few_shot/${location_test}/Llama-3.1-8B/ --out_folder outputs_processed_new/v11_large_2000/sft/few_shot/${location_test} --file_name completion_outputs_llama --fix_missing False


    ########################zero shot###########################
    echo "Processing completion outputs llama(deepseek_llama zero shot)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/${location_test}/DeepSeek-R1-Distill-Llama-8B/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama --fix_missing False

    ########################few shot###########################
    echo "Processing completion outputs llama(deepseek_llama few shot)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/few_shot/${location_test}/DeepSeek-R1-Distill-Llama-8B/ --out_folder outputs_processed_new/v11_large_2000/sft/few_shot/${location_test} --file_name completion_outputs_deepseek_llama --fix_missing False

    ########################CoT###########################
    echo "Processing completion outputs llama(deepseek_llama zero shot CoT)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/${location_test}/DeepSeek-R1-Distill-Llama-8B-CoT/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama_CoT --fix_missing False


    ########################sigspatial###########################
    echo "Processing completion outputs llama(deepseek_llama zero shot sigspatial)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/${location_test}/DeepSeek-R1-Distill-Llama-8B-trained_sigspatial_revise/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama_trained_sigspatial_2000_revise --fix_missing False

    # ############################sft###########################
    echo "Processing completion outputs llama(deepseek_llama sft zero shot)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/${location_test}/DeepSeek-R1-Distill-Llama-8B-trained/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama_trained --fix_missing False

    ########################sft_enhance###########################
    echo "Processing completion outputs llama(deepseek_llama zero shot sft enhance)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/${location_test}/DeepSeek-R1-Distill-Llama-8B-trained_enhance/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama_trained_enhance --fix_missing False
        
done



##########################cross-city###########################
LOCATION_TEST=("DFW" "LA")  #

for location_test in "${LOCATION_TEST[@]}"; do

    # ########################zero shot###########################
    echo "Processing completion outputs llama(deepseek_llama zero shot)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/cross/${location_test}/DeepSeek-R1-Distill-Llama-8B/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama --fix_missing False

    ########################few shot###########################
    echo "Processing completion outputs llama(deepseek_llama few shot)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/few_shot/${location_test}/DeepSeek-R1-Distill-Llama-8B/ --out_folder outputs_processed_new/v11_large_2000/sft/few_shot/${location_test} --file_name completion_outputs_deepseek_llama --fix_missing False

    # ########################CoT###########################
    echo "Processing completion outputs llama(deepseek_llama zero shot CoT)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/cross/${location_test}/DeepSeek-R1-Distill-Llama-8B-CoT/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama_CoT --fix_missing False

    ########################sigspatial###########################
    echo "Processing completion outputs llama(deepseek_llama zero shot sigspatial)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/cross/${location_test}/DeepSeek-R1-Distill-Llama-8B-trained_sigspatial_3000/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama_trained_sigspatial --fix_missing False

    # # ############################sft###########################
    echo "Processing completion outputs llama(deepseek_llama sft zero shot)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/${location_test}/DeepSeek-R1-Distill-Llama-8B-trained/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama_trained --fix_missing False

    # ########################sft_enhance###########################
    echo "Processing completion outputs llama(deepseek_llama zero shot sft enhance)"
    python process_output_individual.py --type completion --in_folder outputs/sft/v11_large_2000/zero_shot/cross/${location_test}/DeepSeek-R1-Distill-Llama-8B-trained_enhance/ --out_folder outputs_processed_new/v11_large_2000/sft/zero_shot/${location_test} --file_name completion_outputs_deepseek_llama_trained_enhance --fix_missing False
        
done


# DeepSeek-R1-Distill-Llama-8B-GRPO8-lm2