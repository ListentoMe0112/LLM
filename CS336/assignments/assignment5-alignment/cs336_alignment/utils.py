from typing import Dict, Callable, Tuple, Literal
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import vllm
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
    ret["input_token_len"] = []

    for i, prompt_str in enumerate(prompt_strs):
        prompt_token = tokenizer.encode(prompt_str)
        output_token = tokenizer.encode(output_strs[i])
        prompt_tensor = torch.tensor(prompt_token)
        output_tensor = torch.tensor(output_token)
        token = torch.cat([prompt_tensor, output_tensor])
        input_id = token[:-1]
        label = token[1:] 
        mask = torch.ones_like(label)
        mask[0:len(prompt_token)-1] = 0
        ret["input_ids"].append(input_id)
        ret["labels"].append(label)
        ret["response_mask"].append(mask)
        ret["input_token_len"].append(len(prompt_token))

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
    ret["input_token_len"] = ret["input_token_len"]
    ret["max_len"] = max_len
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
        min_tokens=4,
        stop=["</answer>"], 
        include_stop_str_in_output=True,
        logprobs=1
    )

    outputs = vllm_model.generate(prompt_strs, sampling_params)

    responses = [out.outputs[0].text for out in outputs]
    logps = [out.outputs[0].logprobs for out in outputs]

    logs = []
    correct_lens = []
    incorrect_lens = []
    for p, a, r, logp in zip(prompt_strs, answers, responses, logps):
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
            "logp" : [v.logprob for t in logp for _, v in t.items()] 
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
    
def answer_transform(ans: str) -> str:
    # Extract the final number from the answer and format it
    import re
    # Look for the final answer number (usually after ####)
    match = re.search(r'####\s*(\d+)', ans)
    if match:
        final_answer = match.group(1)
        # Remove the #### part and keep the reasoning
        reasoning = re.sub(r'####\s*\d+', '', ans).strip()
        formatted_answer = f"{reasoning}\n</think> <answer>{final_answer}</answer>"
    else:
        # If no #### found, try to extract the last number
        numbers = re.findall(r'\d+', ans)
        if numbers:
            final_answer = numbers[-1]
            formatted_answer = f"{ans}\n<answer>{final_answer}</answer>"
        else:
            formatted_answer = ans  # Keep original if no number found

    return formatted_answer

def ground_truth_transform(ans: str) -> float:
    # Extract the final number from the answer and format it
    import re
    # Look for the final answer number (usually after ####)
    match = re.search(r'####\s*(\d+)', ans)
    if match:
        final_answer = match.group(1)
        return final_answer
    else:
        return 0

def compute_group_normalized_rewards(
    reward_fn : Callable[[str, str], dict[str, float]],
    rollout_responses : list[str],
    repeated_ground_truths : list[str],
    group_size : int,
    advantage_eps : float,
    normalize_by_std : bool,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """_summary_
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
            response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    rollout_batch_size = len(rollout_responses) // group_size
    # Create independent sublists to avoid shallow copy issues
    rwds = [[0 for _ in range(group_size)] for _ in range(rollout_batch_size)]
    for i in range(rollout_batch_size):
        for j in range(group_size):
            rwd = reward_fn(rollout_responses[i * group_size + j], repeated_ground_truths[i * group_size + j])
            rwds[i][j] = rwd["reward"]
    raw_rwds = torch.tensor(rwds)
    adv = raw_rwds - raw_rwds.mean(dim = -1, keepdim=True)
    if normalize_by_std:
        adv = adv / (raw_rwds.std(dim= -1, keepdim=True) + advantage_eps)
    
    return adv.reshape(-1), raw_rwds.reshape(-1), dict()

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
            reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
            each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
            be aggregated across the batch and sequence dimensions in the training loop).
    """

    return -policy_log_probs * raw_rewards_or_advantages

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
            probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
            from the old policy.
        cliprange: float Clip parameter ϵ (e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
                loss.
        metadata dict containing whatever you want to log. We suggest logging whether each
            token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
            the min was lower than the LHS.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    advantages_expanded = advantages.expand_as(policy_log_probs)
    surrogate1 = ratio * advantages_expanded
    surrogate2 = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages_expanded
    loss = -torch.minimum(surrogate1, surrogate2)
    metadata = dict()
    # Track whether clipping occurred (i.e., whether surrogate2 was chosen over surrogate1)
    # This happens when ratio is outside [1-cliprange, 1+cliprange]
    clipped = torch.abs(ratio - 1.0) > cliprange
    metadata["clipped"] = clipped
    metadata["clip_ratio"] = clipped.float().mean()  # Fraction of tokens that were clipped
    
    # Log statistics about the ratio
    metadata["ratio_mean"] = ratio.mean()
    metadata["ratio_std"] = ratio.std()
    metadata["ratio_min"] = ratio.min()
    metadata["ratio_max"] = ratio.max()
    
    # Log advantage statistics
    metadata["advantages_mean"] = advantages.mean()
    metadata["advantages_std"] = advantages.std()
    
    # Log the two surrogate values for debugging
    metadata["surrogate1_mean"] = surrogate1.mean()
    metadata["surrogate2_mean"] = surrogate2.mean()
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
            raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
            advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
            (batch_size, 1).
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
            (batch_size, 1).
        old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for "grpo_clip"; scalar ϵ used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        assert raw_rewards != None, "no_baseline needs raw_rewards"
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), dict()
    elif loss_type == "reinforce_with_baseline":
        assert advantages != None, "reinforce_with_baseline needs advantages"
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), dict()
    elif loss_type == "grpo_clip":
        assert advantages != None, "grpo_clip needs advantages"
        assert old_log_probs != None, "grpo_clip needs old_log_probs"
        assert cliprange != None, "grpo_clip needs cliprange"
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all
            masked elements.
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    tensor_masked = tensor * mask
    if dim == None:
        _sum = tensor_masked.sum()
        _count = mask.sum()
        if _count == 0:
            return torch.tensor(float('nan'))
        return _sum / _count
    else:
        _sum = tensor_masked.sum(dim = dim, keepdim = True)
        _count = mask.sum(dim = dim, keepdim= True)
        epsilon = 1e-8
        mean_masked = _sum / (_count + epsilon)
        mean_masked = mean_masked.squeeze(dim)
        invalid_mask = mean_masked == 0
        mean_masked[invalid_mask] = float("nan")

        return mean_masked

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
        loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
    raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
    old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
    cliprange Clip parameter ϵ for GRPO-Clip.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    Implementation tips:
        You should call loss.backward() in this function. Make sure to adjust for gradient
    accumulation.
    """
    loss, meta = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    seq_loss = masked_mean(loss, response_mask, dim=-1)
    seq_loss = seq_loss[~torch.isnan(seq_loss)]
    loss = seq_loss.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, meta
