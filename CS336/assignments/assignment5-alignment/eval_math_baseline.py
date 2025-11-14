from datasets import load_dataset
from vllm import LLM, SamplingParams
from cs336_alignment import utils
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
    acc_cnt = 0
    all_cnt = len(prompts)
    total_reward = 0
    format_reward = 0
    answer_reward = 0

    format_right_and_answer_right = 0
    format_right_and_answer_wrong = 0
    format_wrong_and_answer_wrong = 0
    
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
        else:
            format_wrong_and_answer_wrong += 1

        total_reward += _reward["reward"]
        format_reward += _reward["format_reward"]
        answer_reward += _reward["answer_reward"]
    
    return acc_cnt, all_cnt, format_right_and_answer_right, format_right_and_answer_wrong, format_wrong_and_answer_wrong


if __name__ == "__main__":
    # Load dataset (specify train split)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", dtype="bfloat16")
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
        print(prompt_template.format(question = question["question"]))
        print(utils.answer_transform(question["answer"]))
        prompts.append(prompt_template.format(question = question["question"]))
        ground_truths.append(question["answer"])
        cnt += 1
        
        # Process batch when it reaches 1024 samples
        if cnt >= 1024:
            info = evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)
            acc_cnt_total += info[0]
            all_cnt_total += info[1]
            frar_cnt += info[2]
            fraw_cnt += info[3]
            fwaw_cnt += info[4]
            cnt = 0 
            prompts = []
            ground_truths = []
    
    # Process remaining samples
    if cnt > 0:
        info = evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)
        acc_cnt_total += info[0]
        all_cnt_total += info[1]
        frar_cnt += info[2]
        fraw_cnt += info[3]
        fwaw_cnt += info[4]
    
    # Calculate final accuracy and save results
    accuracy = float(acc_cnt_total) / float(all_cnt_total)
    with open("baseline_eval", "w") as f:
        json.dump({
                "name": "Baseline",
                "accuracy": accuracy, 
                "total_samples": all_cnt_total,
                "correct_samples": acc_cnt_total,
                "format_right_and_answer_right": frar_cnt, 
                "format_right_and_answer_wrong": fraw_cnt, 
                "format_wrong_and_answer_wrong": fwaw_cnt,
                "format_correct_total": frar_cnt + fraw_cnt,
                "answer_correct_total": frar_cnt
                }, f, indent=2)
    print(f"Accuracy: {accuracy:.4f} ({acc_cnt_total}/{all_cnt_total})")
    print(f"Format right & answer right: {frar_cnt}")
    print(f"Format right & answer wrong: {fraw_cnt}")
    print(f"Format wrong & answer wrong: {fwaw_cnt}")
