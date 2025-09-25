import os
import json
from datetime import date
from model_inference import (
    conduct_completion_deepseek_Llama_8B,
)

from tqdm import tqdm, trange

from vllm import LLM, SamplingParams
import google.generativeai as genai
from utils import (
    is_complete_table_output
)
import argparse
from prompt import few_shot_examples_v11, system_prompt_v111,suffix_v111
import setproctitle

setproctitle.setproctitle('@liupj')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Similate a survey using LLMs.'
    )
    parser.add_argument(
        '--local_rank',
    )
    parser.add_argument(
        '--type',
        dest='type',
        type=str,
        choices=[
            'survey',
            'completion'
        ],
        default='completion'
    )
    parser.add_argument(
        '--location',
        dest='location',
        type=str,
        choices=[
            'sf',
            'dc',
            'dfw',
            'minneapolis',
            'la'
        ]
    )

    parser.add_argument(
        '--travel_file',
        dest='travel_file',
        type=str,
        choices=[
            'travel',
            'travel2',
            'travel3'
        ],
        default='travel'
    )

    parser.add_argument(
        '--loc_ub',
        dest='loc_ub',
        type=str,
        choices=[
            'downtown',
            'suburb',
            'none'
        ],
        default='none'
    )

    parser.add_argument(
        '--use_popn_sampling',
        dest='use_popn_sampling',
        type=str,
        choices=[
            'true',
            'false',
        ],
        default='true'
    )

    parser.add_argument(
        '--use_date_sampling',
        dest='use_date_sampling',
        type=str,
        choices=[
            'true',
            'false',
        ],
        default='true'
    )

    parser.add_argument(
        '--use_model',
        dest='use_model',
        type=str,
        choices=[
            'DeepSeek-R1-Distill-Llama-8B',
            'Llama-3.1-8B',
            'DeepSeek-R1-Distill-Llama-8B-trained',
            "DeepSeek-R1-Distill-Llama-8B-trained_enhance"
        ],
        default='llama'
    )

    parser.add_argument(
        '--out_folder',
        dest='out_folder',
        type=str,
        default='outputs'
    )

    parser.add_argument(
        '--fix_bias',
        dest='fix_bias',
        type=str,
        default='false',
        choices=[
            'false',
            '1',
            '2',
            'covid'
        ]
    )
    parser.add_argument(
        '--year',
        dest='year',
        type=str,
        default='2021',
        choices=[
            '2017',
        ]
    )
    parser.add_argument(
        '--trained_model_epoch',
        dest='trained_model_epoch',
        type=int,
        default=1,
        choices=[
            1,
            3,
            5,
            10,
            20
        ]
    )
    parser.add_argument(
        '--trained_db_size',
        dest='trained_sb_size',
        type=str,
        default=None,
        choices=[
            None,
            '1000',
            '10000',
            '10000-exclude-inf-cities',
        ]
    )

    parser.add_argument(
        '--few_shot',
        dest='few_shot',
        type=str,
        choices=[
            'true',
            'false',
        ],
        default='false'
    )
    parser.add_argument(
        '--profile_type',
        dest='profile_type',
        type=str,
        choices=[
            'actual',
            'sample',
        ],
        default='sample'
    )
    parser.add_argument(
        '--location_test',
        dest='location_test',
        type=str,
        choices=[
            'SF',
            'SD',
            'ATL',
            'ATX'
        ],
        default='sample'
    )

    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5" 
    
    quantization = True
    if args.use_model == 'Llama-3.1-8B':
        model_mode='basic'
        model_type='base'
    elif  args.use_model == 'DeepSeek-R1-Distill-Llama-8B':
        model_mode='basic'
        model_type='reasoning'
    elif args.use_model == 'Llama-3.1-8B-trained':
        model_mode='basic'
        model_type='base'
    elif args.use_model == 'DeepSeek-R1-Distill-Llama-8B-trained' or args.use_model == 'DeepSeek-R1-Distill-Llama-8B-trained_enhance':
        model_mode='basic'
        model_type='reasoning'


    if model_mode == 'basic':
        if args.use_model == 'Llama-3.1-8B':
            model_name = 'LLaMA-Factory/model/basic/meta-llama/Llama-3.1-8B'
        elif args.use_model == 'DeepSeek-R1-Distill-Llama-8B':
            model_name = 'LLaMA-Factory/model/basic/deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
        
        elif args.use_model == 'Llama-3.1-8B-trained':
            model_name = 'LLaMA-Factory/model/lora/v11_large_2000/Llama-3.1-8B-trained'
        elif args.use_model == 'DeepSeek-R1-Distill-Llama-8B-trained':
            model_name = 'LLaMA-Factory/model/lora/v11_large_2000/DeepSeek-R1-Distill-Llama-8B-trained'
        elif args.use_model == 'DeepSeek-R1-Distill-Llama-8B-trained_enhance':
            model_name = 'LLaMA-Factory/model/lora/v11_large_2000/DeepSeek-R1-Distill-Llama-8B-trained_enhance'
       


        if model_type == 'base':
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=1,
                top_k=50,
                max_tokens=1024,
                repetition_penalty=1.0,
                # length_penalty=1.0,
            )
        elif model_type == 'reasoning':
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=1,
                top_k=50,
                max_tokens=4096,
                repetition_penalty=1.0,
                # length_penalty=1.0,
            )



    test_data_path=f'training_datasets/train_data_v1_1/new_data/test/{args.location_test}_constructed_examples_v11_test_gt_select.json'
    if args.profile_type == 'actual':
        with open(test_data_path, 'r') as file:                 ###training_datasets/train_data_v1_1/SF_constructed_examples_v11_test.json
            data = json.load(file)
        if model_mode != 'api':
            llm = LLM(model_name, tensor_parallel_size=2)

        for key, value in tqdm(data.items(), desc="Processing", unit="item"):
            output_path = f"{args.out_folder}/completions_{args.location}_{key}.json"
            if os.path.exists(output_path):
                continue
            if args.few_shot == 'true':
                prompt = system_prompt_v111 + '\n' + few_shot_examples_v11 + 'Task:\n' + value['profiles'] + suffix_v111
            elif args.few_shot == 'false':
                prompt = system_prompt_v111 + 'Task:\n' + value['profiles'] + suffix_v111


            if args.use_model in [
                'DeepSeek-R1-Distill-Llama-8B',
                'DeepSeek-R1-Distill-Llama-8B-trained',
                'DeepSeek-R1-Distill-Llama-8B-trained_enhance',

            ]:
                prompt = prompt + '<think>\n'
            # print('1')
            if args.type == 'completion':
                for attempt in range(1, 4):
                    if args.use_model in [
                         'Llama-3.1-8B',
                        'DeepSeek-R1-Distill-Llama-8B',
                        'DeepSeek-R1-Distill-Llama-8B-trained',
                        'Llama-3.1-8B-trained',
                        "DeepSeek-R1-Distill-Llama-8B-trained_enhance"
                    ]:
                        answers = conduct_completion_deepseek_Llama_8B(
                            llm,
                            sampling_params=sampling_params,
                            prompt_file=prompt,
                        )
                    # else:
                    #     answers = conduct_completion(
                    #         model,
                    #         tokenizer,
                    #         prompt_file=prompt,
                    #     )

                    ans_output = answers.get("ans_output", "")
                    is_complete = is_complete_table_output(ans_output)

                    answers.update({
                        'key': key,
                        'location': args.location,
                        'model': args.use_model,
                        'loc_type': value['loc_type'],
                        'travel_times': value['travel_times'],
                        'is_complete_table': is_complete,
                        'generation_attempt': attempt
                    })

                    if is_complete or attempt == 3:
                        json.dump(
                            answers,
                            open(
                                f'{args.out_folder}/completions_{args.location}_{key}.json',
                                'w+'
                            )
                        )
                        break

