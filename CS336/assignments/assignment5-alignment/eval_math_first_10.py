from datasets import load_dataset
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import Callable, List, Dict, Tuple
import json

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams
) -> Tuple[int, int, int, int, int]:
    """
    Evaluate VLLM model performance on math problems
    
    Args:
        vllm_model: The VLLM model to evaluate
        reward_fn: Function to compute reward score
        prompts: List of input prompts
        ground_truths: List of ground truth answers
        eval_sampling_params: Sampling parameters for generation
    
    Returns:
        Tuple containing: (acc_cnt, all_cnt, format_right_and_answer_right, 
                          format_right_and_answer_wrong, format_wrong_and_answer_wrong)
    """
    all_cnt = len(prompts)

    format_right_and_answer_right = 0
    format_right_and_answer_wrong = 0
    format_wrong_and_answer_wrong = 0
    first_10_fwaw_idx = []
    fwawoutput = []
    first_10_fraw_idx = []
    frawoutput = []
    
    # Generate outputs for all prompts
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # Evaluate each output
    for i, output in enumerate(outputs):
        # Extract generated text from output
        generated_text = output.outputs[0].text
        _reward = reward_fn(generated_text, ground_truths[i])
        
        if _reward["reward"] >= 1.0:
            acc_cnt += 1

        if _reward["format_reward"] > 0 and _reward["answer_reward"] > 0:
            format_right_and_answer_right += 1
        elif _reward["format_reward"] > 0:
            format_right_and_answer_wrong += 1 
            first_10_fraw_idx.append(i)
            frawoutput.append(output)
        else:
            format_wrong_and_answer_wrong += 1
            first_10_fwaw_idx.append(i)
            fwawoutput.append(output)

        if len(first_10_fraw_idx) >= 10 and len(first_10_fwaw_idx) >= 0:
            break
    
    return fwawoutput, frawoutput


if __name__ == "__main__":
    # Load dataset (specify train split)
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )
    
    # Load prompt template
    with open("./cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()
    
    # Prepare prompts and ground truths
    prompts = []
    ground_truths = []
    cnt = 0
    acc_cnt_total = 0
    all_cnt_total = 0
    frar_cnt = 0
    fraw_cnt = 0
    fwaw_cnt = 0
    
    # Process dataset in batches
    for idx, question in enumerate(ds):
        print(prompt_template.format(question = question["problem"]))
        print(question["answer"])
        prompts.append(prompt_template.format(question = question["problem"]))
        ground_truths.append(question["answer"])
        cnt += 1
        
        # Process batch when it reaches 1024 samples
        if cnt >= 1024:
            info = evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)
            print("--------------------------- first 10 fwaw ------------------------------------")
            print(info[0])
            print("--------------------------- first 10 fwaw ------------------------------------")

            print("--------------------------- first 10 fraw ------------------------------------")
            print(info[1])
            print("--------------------------- first 10 fraw ------------------------------------")

