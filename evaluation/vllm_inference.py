"""Generate answers with local models.

Usage:
python3 gen_model_answer_01.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import ray
import argparse
import json
import os
import random
import re
import time
from tensor_parallel import TensorParallelPreTrainedModel
import shortuuid
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from common import load_questions, temperature_config, get_all_filenames
from model import load_model, get_conversation_template
from common import should_process_file
from vllm import LLM, SamplingParams
from tensor_parallel import TensorParallelPreTrainedModel


def process_file(model_id, questions, answer_file, max_new_token, num_choices, data_id, model, tokenizer, max_token):
    prompt_dict = {}
    prompts = []
    question_id = 0

    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)

            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]

                conv.append_message(conv.roles[0], qs)

                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                prompt_dict.update({question_id: [prompt, qs]})
                prompts.append(prompt)
        question_id += 1
    try:
        sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=max_token,repetition_penalty=1.2)
        output_ids = model.generate(
            prompts,
            sampling_params,
        )
        # output = output_ids[0].outputs[0].text
    except RuntimeError as e:
        print(e)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    for question_id, output in enumerate(output_ids):
        output = output.outputs[0].text

        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]

        if conv.name == "xgen" and output.startswith("Assistant:"):
            output = output.replace("Assistant:", "", 1).strip()
        output = output.split("</s>")[0]
        if "label" in model_id:
            # if "baichuan" in model_id:
            #     use_prompt = prompt_dict[question_id][40:-14]
            # else:
                # use_prompt=prompt_dict[question_id][38:-14]
            use_prompt = prompt_dict[question_id]
            with open(os.path.expanduser(answer_file + data_id), "a", encoding='utf-8') as fout:
                ans_json = {
                    "dataType": questions[question_id]["dataType"],
                    "question_id": question_id,
                    "instruction": output,
                    "output": use_prompt[1],
                    "use_prompt": use_prompt[0],
                }
                json.dump(ans_json, fout, ensure_ascii=False)
                fout.write("\n")
        elif "point" in model_id or "stem" in model_id:
            with open(os.path.expanduser(answer_file + data_id), "a", encoding='utf-8') as fout:

                ans_json = {
                    "score": output,
                    "dataType": questions[question_id]["dataType"],
                    "question_id": question_id,
                    "instruction": questions[question_id]["instruction"],
                    "output": questions[question_id]["output"],
                    "use_prompt": prompt_dict[question_id][0],
                }
                json.dump(ans_json, fout, ensure_ascii=False)
                fout.write("\n")
        else:
            with open(os.path.expanduser(answer_file + data_id), "a", encoding='utf-8') as fout:
                ans_json = {
                    "instruction": questions[question_id]["instruction"],
                    "input": "",
                    "output": output,
                    "dataType": questions[question_id]["dataType"],
                    "question_id": question_id,
                    "use_prompt": prompt_dict[question_id][0],
                }
                json.dump(ans_json, fout, ensure_ascii=False)
                fout.write("\n")


if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="The path to the data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="The path to save result",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )

    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--max_token",
        type=int,
        default=512,
        help="output length",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=8, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    args = parser.parse_args()
    model = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=1)

    filenames = get_all_filenames(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # import pdb; pdb.set_trace()
    filenames.reverse()
    for i, filename in enumerate(filenames):
        print(
            f"------------------------------------now is the {i}/{len(filenames)}---{filename}---------------------------------")
        if os.path.exists(args.output_path + filename):
            print("exits.continue")
            continue
        questions = load_questions(args.data_path + filename, args.question_begin, args.question_end, tokenizer,
                                   args.model_id)

        process_file(args.model_id, questions, args.output_path, args.max_new_token, args.num_choices, filename, model,
                     tokenizer, args.max_token)
