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
    global_seed =42
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    lr = 5e-6
    gradient_accumulation_steps = 16
    micro_batch_size = 2
    num_epochs = 1
    log_every = 50
    eval_every = 200
    
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

    vllm_model = init_vllm(model_id, "cuda:0", global_seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    policy.train()

    def collate_fn(batch):
        prompts = [prompt_template.format(question = ex["question"]) for ex in batch ] 
        # Format answers to match the template
        formatted_answers = []
        ground_truths = []
        for ans in [ex["answer"] for ex in batch]:
            formatted_answer = utils.answer_transform(ans)
            formatted_answers.append(formatted_answer)
            ground_truth = utils.answer_transform(ans)
            ground_truths.append(ground_truth)
        return {"prompts" : prompts,  "answers" : formatted_answers, "ground_truths" : ground_truths}

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size = 64,
        collate_fn=collate_fn
    )

    step = 0
    for epoch in range(num_epochs):
        for batch in train_loader:
            prompts = batch["prompts"]
            answers = batch["answers"]
            input_infos = utils.tokenize_prompt_and_output(prompt_strs=prompts, output_strs=answers, tokenizer=tokenizer)
            ret = utils.get_response_log_probs(policy, input_infos["input_ids"].to("cuda:1"), input_infos["labels"].to("cuda:1"))
            loss, _ = utils.sft_microbatch_train_step(ret["log_probs"], input_infos["response_mask"].to("cuda:1"), 16)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % log_every == 0:
                print(f"step {step}  loss={loss.item():.4f}")

            # if (step + 1) % eval_every == 0:
            #     evaluate(val_loader, tokenizer, vllm_model, policy)

            step += 1

        # ---------- 一个 epoch 结束后，完整评测验证集 ----------
        evaluate(val_loader, tokenizer, vllm_model, policy)
        
        # ---------- Save checkpoint ----------
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        policy.save_pretrained(checkpoint_path, safe_serialization=True)
        
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save optimizer state
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save training metadata
        metadata = {
            "epoch": epoch + 1,
            "step": step,
            "global_seed": global_seed,
            "model_id": model_id,
            "lr": lr,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "micro_batch_size": micro_batch_size,
        }
        metadata_path = os.path.join(checkpoint_path, "training_metadata.json")
        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_path}")
