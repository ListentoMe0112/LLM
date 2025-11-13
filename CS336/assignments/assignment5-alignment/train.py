from cs336_alignment import utils
import torch
from datasets import load_dataset
from vllm.model_executor import set_random_seed as vllm_set_random_seed


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
            dtype=torch.float32,
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
    raw_ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    train_val = raw_ds.train_test_split(test_size=0.05, seed=42)
    train_ds = train_val["train"]
    val_ds   = train_val["test"]
    return train_ds, val_ds

@torch.no_grad()
def evaluate(val_loader, tokenizer, vllm_model):
    """遍历验证集，用 vLLM 生成并打印最终指标"""
    all_logs = []
    for batch in tqdm(val_loader, desc="Eval"):
        prompts = batch["prompts"]
        answers = batch["answers"]
        logs = utils.log_generations_vllm(
            prompts,
            answers,
            tokenizer,
            vllm_model,
            reward_fn=utils.dummy_reward,  # 如果换了 reward 模型，这里改掉
        )
        all_logs.extend(logs)

    # 简单聚合
    total = len(all_logs)
    correct = sum(int(l["answer_reward"] > 0) for l in all_logs)
    avg_len = sum(l["response_length"] for l in all_logs) / total
    print(f"Eval  finish  |  total={total}  correct={correct}  acc={correct/total:.4f}  avg_len={avg_len:.2f}")

if __name__ == "__main__":
    global_seed =42
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    lr = 5e-6
    gradient_accumulation_steps = 8
    micro_batch_size = 4
    num_epochs = 1
    log_every = 50

    # Load prompt template
    with open("./cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    torch.manual_seed(global_seed)

    train_ds, val_ds = init_datasets() 

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cuda:0",
    )

    vllm_model = init_vllm(model_id, "cuda:1", 42)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    policy.train()

    def collate_fn(batch):
        prompts = [prompt_template.format(question = ex["problem"]) for ex in batch ] 
        solutions = [ex["solution"] for ex in batch ] 
        answers = [ex["answer"] for ex in batch]
        return {"prompts" : prompts, "solutions" : solutions, "answers" :answers}

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
            solutions = batch["solutions"]
            answers = batch["answers"]
            input_infos = utils.tokenize_prompt_and_output(prompt_strs=prompts, output_strs=solutions, tokenizer=tokenizer)
            ret = utils.get_response_log_probs(policy, input_infos["input_ids"], input_infos["labels"])
            loss, _ = utils.sft_microbatch_train_step(ret["log_probs"], input_infos["response_mask"], 8)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % log_every == 0:
                print(f"step {step}  loss={loss.item():.4f}")

            if step % eval_every == 0:
                evaluate(val_loader, tokenizer, vllm_model)

            step += 1

        # ---------- 一个 epoch 结束后，完整评测验证集 ----------
        evaluate(val_loader, tokenizer, vllm_model)