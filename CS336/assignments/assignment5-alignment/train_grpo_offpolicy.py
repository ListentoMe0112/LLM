import os
from cs336_alignment import utils
import torch
import torch.distributed as dist
from datasets import load_dataset
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedTokenizerBase, PreTrainedModel,  AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from unittest.mock import patch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def init_datasets():
    raw_ds_train = load_dataset("openai/gsm8k", "main",split="train")
    raw_ds_val = load_dataset("openai/gsm8k", "main",split="test")
    return raw_ds_train,raw_ds_val 

@torch.no_grad()
def evaluate(val_loader, tokenizer, vllm_model, policy):
    """遍历验证集，用 vLLM 生成并打印最终指标"""
    load_policy_into_vllm_instance(policy, vllm_model)
    all_logs = []
    for batch in tqdm(val_loader, desc="Eval"):
        prompts = batch["prompts"]
        ground_truths = batch["ground_truths"]
        logs = utils.log_generations_vllm(
            prompts,
            ground_truths,
            tokenizer,
            vllm_model,
            reward_fn=r1_zero_reward_fn,  
        )
        all_logs.extend(logs)

    # 简单聚合
    total = len(all_logs)
    correct = sum(int(l["answer_reward"] > 0) for l in all_logs)
    avg_len = sum(l["response_length"] for l in all_logs) / total
    print(f"Eval  finish  |  total={total}  correct={correct}  acc={correct/total:.4f}  avg_len={avg_len:.2f}")
    print(all_logs[:10])

if __name__ == "__main__":
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size = 8
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal["no_baseline", "reinforce_with_baseline","grpo_clip"] = "grpo_clip"
    use_std_normalization: bool = True

    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps # 2
    assert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size # 32
    assert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )

    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size # 128


    global_seed =42
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    log_every = 32 
    eval_every = 10
    
    # Create checkpoint directory
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load prompt template
    with open("./cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    torch.manual_seed(global_seed)

    train_ds, val_ds = init_datasets() 

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1",
    )

    policy.train()

    vllm_model = init_vllm(model_id, "cuda:0", global_seed)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    def collate_fn(batch):
        prompts = [prompt_template.format(question = ex["question"]) for ex in batch ] 
        # Format answers to match the template
        formatted_answers = []
        ground_truths = []
        for ans in [ex["answer"] for ex in batch]:
            formatted_answer = utils.answer_transform(ans)
            formatted_answers.append(formatted_answer)
            ground_truth = utils.ground_truth_transform(ans)
            ground_truths.append(ground_truth)
        return {"prompts" : prompts,  "answers" : formatted_answers, "ground_truths" : ground_truths}

    train_loader = DataLoader(
        train_ds,
        batch_size=n_prompts_per_rollout_batch,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size = 64,
        collate_fn=collate_fn
    )

    train_iter = iter(train_loader)
    for grpo_step in range(n_grpo_steps):
        batch = next(train_iter) 
        prompts = batch["prompts"] # 32 prompts
        ground_truths = batch["ground_truths"] # 32 ground_truths
        # Expand ground truths for group_size
        repeated_ground_truths = []
        repeated_prompts = []

        for gt in ground_truths:
            repeated_ground_truths.extend([gt] * group_size)
        for prompt in prompts:
            repeated_prompts.extend([prompt] * group_size)

        # generate vllm outcome
        with torch.no_grad():
            load_policy_into_vllm_instance(policy, vllm_model)
            responses = []
            logs = utils.log_generations_vllm(
                repeated_prompts,
                repeated_ground_truths,
                tokenizer,
                vllm_model,
                reward_fn=r1_zero_reward_fn,  
            )
                
            responses = [log["generated_response"]for log in logs]
            # logps = [log["logp"] for log in logs ]
            assert len(responses) == rollout_batch_size, f"expected length {rollout_batch_size}, get {len(responses)}"

        # calculate oldlogp
        with torch.no_grad():
            ret = utils.tokenize_prompt_and_output(repeated_prompts, responses, tokenizer)
            responses_mask = ret["response_mask"].to("cuda:1")
            old_logps = []
            for i in range(0, len(responses_mask), micro_train_batch_size):
                old_logps_ret = utils.get_response_log_probs(policy, ret["input_ids"][i:i+micro_train_batch_size].to("cuda:1"), ret["labels"][i:i+micro_train_batch_size].to("cuda:1"), False)
                old_logps.extend(old_logps_ret["log_probs"])
            old_logps = torch.stack(old_logps, device= old_logps[0].device, dtype = old_logps[0].dtype)

        advantages, raw_rewards, _ = utils.compute_group_normalized_rewards(
            r1_zero_reward_fn, 
            responses, 
            repeated_ground_truths, 
            group_size, 
            advantage_eps, 
            False
        ) 
        # 默认在cpu上，这里需要转到cuda:1上
        advantages = advantages.to("cuda:1")
        raw_rewards = raw_rewards.to("cuda:1")

        total_train_step = rollout_batch_size  * epochs_per_rollout_batch // micro_train_batch_size
        for train_step in range(total_train_step):
            start_idx = (train_step * micro_train_batch_size) % rollout_batch_size
            end_idx = start_idx + micro_train_batch_size

            micro_input_ids = ret["input_ids"][start_idx : end_idx].to("cuda:1")
            micro_input_labels = ret["labels"][start_idx : end_idx].to("cuda:1")
            micro_response_mask = ret["response_mask"][start_idx : end_idx].to("cuda:1")
            micro_old_log_probs = old_logps[start_idx : end_idx].to("cuda:1")
            micor_logp_ret = utils.get_response_log_probs(policy, micro_input_ids, micro_input_labels, False)
            micro_log_probs = micor_logp_ret["log_probs"].to("cuda:1")
            micro_advantages = advantages[start_idx : end_idx]
            micro_raw_rewards = raw_rewards[start_idx : end_idx]

            if train_step == 0:
                assert torch.allclose(utils.masked_mean(micro_log_probs, micro_response_mask, dim = -1) , utils.masked_mean(micro_old_log_probs, micro_response_mask, dim = -1), atol=1e-4 ), f"exptect{micro_log_probs} get {micro_response_mask}"

            loss, _ = utils.grpo_microbatch_train_step(
                micro_log_probs, 
                micro_response_mask, 
                gradient_accumulation_steps, 
                loss_type, 
                micro_raw_rewards.unsqueeze(-1), 
                micro_advantages.unsqueeze(-1), 
                micro_old_log_probs, 
                0.8 
            )

            if (train_step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (train_step + 1) % log_every == 0:
                print(f"step {grpo_step}  loss={loss.item():.4f}, adv={micro_advantages.mean()}, raw_rewards={micro_raw_rewards.mean()}")
        
        
        if (grpo_step + 1) % eval_every == 0:
            evaluate(val_loader, tokenizer, vllm_model, policy)

