from typing import Dict
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import llm
from vllm import SamplingParams
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer) -> Dict[str, torch.Tensor]: 
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for 
    other tokens (prompt or padding).
    Args:
    prompt_strs: list[str] List of prompt strings.
    output_strs: list[str] List of output strings.
    tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    Returns:
    dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
        input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            the tokenized prompt and output strings, with the final token sliced off.
        labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            shifted input ids, i.e., the input ids without the first token.
        response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -1): 
            a mask on the response tokens in the labels.
    """
    ret = dict() 
    ret["input_ids"] = []
    ret["labels"] = []
    ret["response_mask"] = []

    for i, prompt_str in enumerate(prompt_strs):
        prompt_token = tokenizer.encode(prompt_str)
        output_token = tokenizer.encode(output_strs[i])
        prompt_tensor = torch.tensor(prompt_token)
        output_tensor = torch.tensor(output_token)
        token = torch.cat([prompt_tensor, output_tensor])
        input_id = token[:-1]
        label = token[1:] 
        mask = torch.ones_like(label)
        mask[0:len(prompt_token) - 1] = 0
        ret["input_ids"].append(input_id)
        ret["labels"].append(label)
        ret["response_mask"].append(mask)
        if i == 0:
            print(token)

    max_len = max(len(seq) for seq in ret["input_ids"])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_labels = []
    padded_masks = []

    for i in range(len(ret["input_ids"])):
        input_seq = ret["input_ids"][i]
        label_seq = ret["labels"][i]
        mask_seq = ret["response_mask"][i]

        input_padding = torch.full((max_len - len(input_seq),), pad_token_id, dtype=input_seq.dtype)
        label_padding = torch.full((max_len - len(label_seq),), pad_token_id, dtype=label_seq.dtype)
        mask_padding = torch.zeros(max_len - len(mask_seq), dtype=mask_seq.dtype)
        
        padded_input_ids.append(torch.cat([input_seq, input_padding]))
        padded_labels.append(torch.cat([label_seq, label_padding]))
        padded_masks.append(torch.cat([mask_seq, mask_padding]))
   
    ret["input_ids"] = torch.stack(padded_input_ids)
    ret["labels"] = torch.stack(padded_labels)
    ret["response_mask"] = torch.stack(padded_masks)
    return ret


def compute_entropy(logits: torch.Tensor):
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.
        Returns: torch.Tensor Shape (batch_size, sequence_length). 
        The entropy for each next-token prediction.
    """
    prob = torch.softmax(logits, dim = -1)
    entropy = -torch.sum(prob * torch.log(prob), dim=-1)
    return entropy
   

def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    """
    Implement a method get_response_log_probs that gets per-token conditional
    log-probabilities (given the previous tokens) from a causal language model, and optionally the
    entropy of the model’s next-token distribution.
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
            response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
            compute_entropy.
    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
            log pθ (xt | x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True).
    """
    ret = dict()
    output = model(input_ids)
    logits = model(input_ids).logits #batch_size, seq_len, vocab_size
    prob = torch.softmax(logits, dim = -1) # batch_size, seq_len, vocab_size
    log_prob = torch.log(prob )
    ret["log_probs"] = torch.gather(log_prob, dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        ret["token_entropy"] = compute_entropy(logits)
    return ret

def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
    ) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask
    == 1.
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
            dimensions.
    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
        the sum.
    """
    sum_tensor = torch.sum(mask * tensor, dim = dim)
    normalize_tensor = sum_tensor / normalize_constant
    return normalize_tensor
    
def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
                this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any other statistics you
                might want to log.
    """
    loss = -torch.mean(masked_normalize(policy_log_probs, response_mask, normalize_constant, -1))
    loss = loss / gradient_accumulation_steps
    loss.backward()
    metadata = dict()
    return loss, metadata

def log_generations_vllm(
    prompt_strs: list[str],
    answers: list[str],
    tokenizer: PreTrainedTokenizerBase,
    vllm_model,          # vLLM 的 LLM 实例
    reward_fn: Callable[[str, str], Dict[str, float]]
) -> list[dict]:
    """
    Generate responses with vLLM and collect rich logging info.
    Args:
        prompt_strs: list of input prompts.
        answers: list of ground-truth answers (same order).
        tokenizer: HuggingFace tokenizer.
        vllm_model: vLLM LLM object (already loaded).
        max_new_tokens: generation length limit.
        temperature: sampling temperature.
    Returns:
        list[dict] – one dict per example with all requested fields.
    """

    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"], 
        include_stop_str_in_output=True,
    )

    outputs = vllm_model.generate(prompt_strs, sampling_params)

    responses = [out.outputs[0].text for out in outputs]

    logs = []
    correct_lens = []
    incorrect_lens = []
    for p, a, r in zip(prompt_strs, answers, responses):
        rew = reward_fn(r, a)
        logs.append({
            "prompt": p,
            "generated_response": r,
            "ground_truth_answer": a,
            "format_reward": rew["format_reward"],
            "answer_reward": rew["answer_reward"],
            "reward": rew["reward"],
            "avg_token_entropy": 0.0,  # vLLM 不返回 logits，可后续用 HF 模型补算
            "response_length": len(tokenizer.encode(r)),
        })
        if rew["answer_reward"] > 0:
            correct_lens.append(len(tokenizer.encode(r)))
        else:
            incorrect_lens.append(len(tokenizer.encode(r)))

    avg_len = sum(len(tokenizer.encode(r)) for r in responses) / len(responses)
    avg_correct_len = sum(correct_lens) / max(len(correct_lens), 1)
    avg_incorrect_len = sum(incorrect_lens) / max(len(incorrect_lens), 1)
    for rec in logs:
        rec["avg_response_length"] = avg_len
        rec["avg_correct_response_length"] = avg_correct_len
        rec["avg_incorrect_response_length"] = avg_incorrect_len

    return logs
    


    