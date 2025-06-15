#!/usr/bin/env python3
# FILEPATH: /data/home/yeeboxie/LLM/CS336/assignments/assignment1-basics/train.py

import os
import time
import argparse
import numpy as np
import torch
import wandb
from cs336_basics.utils import (
    TransformerLanguageModel,
    AdamW,
    get_batch,
    cross_entropy,
    gradient_clipping,
    learning_rate_schedule,
    save_checkpoint,
    load_checkpoint
)
from torch.utils.tensorboard.writer import SummaryWriter

writer = SummaryWriter('./pytorch_tb/')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Transformer Language Model Training')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data file')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=1024, help='Context length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=100000, help='Maximum training iterations')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=100000, help='Learning rate decay iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='Evaluation iterations')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps', help='Device to use')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Memory-efficient dataset loading with np.memmap
    print(f"Loading datasets with np.memmap...")
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')
    print(f"Training data size: {len(train_data):,} tokens")
    print(f"Validation data size: {len(val_data):,} tokens")

    # Initialize model
    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        theta=args.rope_theta,
        max_seq_len=args.context_length,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        device=args.device
    ).to(args.device)    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Compile model if requested
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=1e-8
    )

    # Training state
    iter_num = 0
    best_val_loss = float('inf')

    # Training loop
    print(f"Starting training on {args.device}...")
    start_time = time.time()
    
    while iter_num < args.max_iters:
        # Set learning rate for this iteration
        lr = learning_rate_schedule(
            iter_num,
            alpha_max=args.learning_rate,
            alpha_min=args.min_lr,
            Tw=args.warmup_iters,
            Tc=args.lr_decay_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get a batch of data
        x, y = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device
        )

        # Forward pass
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)
        
        # Optimizer step
        optimizer.step()

        # Log training metrics
        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_processed = iter_num * args.batch_size * args.context_length
            tokens_per_sec = tokens_processed / elapsed
            
            print(f"Iter {iter_num:6d} | Loss {loss.item():.4f} | LR {lr:.2e} | TPS {tokens_per_sec/1e6:.2f}M/s")
            writer.add_scalar("train/loss", loss.item(), iter_num)
            writer.add_scalar("train/lr", lr, iter_num)
            writer.add_scalar("train/tokens_per_sec", tokens_per_sec, iter_num)

        # Evaluate on validation set
        if iter_num % args.eval_interval == 0:
            val_losses = []
            model.eval()
            with torch.no_grad():
                for _ in range(args.eval_iters):
                    x_val, y_val = get_batch(
                        dataset=val_data,
                        batch_size=args.batch_size,
                        context_length=args.context_length,
                        device=args.device
                    )
                    logits = model(x_val)
                    val_loss = cross_entropy(logits.view(-1, logits.size(-1)), y_val.view(-1))
                    val_losses.append(val_loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"Validation loss at iter {iter_num}: {avg_val_loss:.4f}")
            
            writer.add_scalar("iter", iter_num, iter_num)
            writer.add_scalar("val/loss", avg_val_loss, iter_num)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(args.out_dir, "best_model.pt")
                save_checkpoint(model, optimizer, iter_num, checkpoint_path)
                print(f"Saved best model to {checkpoint_path} with loss {best_val_loss:.4f}")
            
            model.train()

        # Save checkpoint
        if iter_num % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        iter_num += 1

    print("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()

