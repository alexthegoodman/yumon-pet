import os
import sys
import json
import argparse
import torch
from tokenizers import Tokenizer

from model import YumonPetMoeBrain, ModelConfig
from dataloader import BOS_TOKEN, EOS_TOKEN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config_dict = ckpt.get("config", {})
    model_config = ModelConfig.from_dict(config_dict)
    model_config.vocab_size = tokenizer.get_vocab_size()

    model = YumonPetMoeBrain(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer, model_config


def generate_reply(
    model: YumonPetMoeBrain,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    encoded = tokenizer.encode(prompt, add_special_tokens=False).ids

    # Training format: [BOS_TOKEN] + prompt_ids + [SEP_TOKEN (space)] + reply_ids + [EOS_TOKEN]
    space_ids = tokenizer.encode(" ", add_special_tokens=False).ids
    sep_tokens = space_ids if space_ids else [146]
    prompt_ids = [BOS_TOKEN] + encoded + sep_tokens
    prompt_len = len(prompt_ids)

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = model.generate(
        prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=EOS_TOKEN,
    )

    out_ids = generated[0].tolist()
    reply_ids = out_ids[prompt_len:]
    reply = tokenizer.decode(reply_ids).strip()

    return reply


def main():
    default_tok = "tokenizer.json" if os.path.exists("tokenizer.json") else "../yumon_bpe/tokenizer-16k-2.json"
    parser = argparse.ArgumentParser(description="Generate replies with trained Yumon Pet Sparse MoE Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default=default_tok, help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string (if omitted, interactive mode starts)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)

    if args.prompt:
        reply = generate_reply(
            model, tokenizer, args.prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("\n=== Generation ===")
        print(f"Prompt: {args.prompt}")
        print(f"Reply:  {reply}")
    else:
        print("\nEntering interactive chat mode with Yumon Pet (type 'exit' or 'quit' to quit):")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input or user_input.lower() in ["exit", "quit"]:
                    break
                reply = generate_reply(
                    model, tokenizer, user_input, device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                print(f"Yumon: {reply}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
