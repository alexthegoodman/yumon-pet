import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Sequence, Callable, Iterable
import torch
from tokenizers import Tokenizer

from model import YumonPetMoeBrain, ModelConfig
from dataloader import BOS_TOKEN, EOS_TOKEN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HISTORY_LIMIT = 200


@dataclass
class ChatMessage:
    role: str
    text: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(role=str(data.get("role", "")), text=str(data.get("text", "")))


def load_history(path: str) -> List[ChatMessage]:
    """Load chat history from a JSON file."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages: List[ChatMessage] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                messages.append(ChatMessage.from_dict(item))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                messages.append(ChatMessage(role=str(item[0]), text=str(item[1])))
    if len(messages) > HISTORY_LIMIT:
        messages = messages[-HISTORY_LIMIT:]
    return messages


def save_history(path: str, history: Sequence[Any]) -> None:
    """Save chat history atomically to a JSON file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temp_path = path + ".tmp"
    records = []
    for m in history:
        if isinstance(m, ChatMessage):
            records.append(m.to_dict())
        elif isinstance(m, dict):
            records.append({"role": str(m.get("role", "")), "text": str(m.get("text", ""))})
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            records.append({"role": str(m[0]), "text": str(m[1])})
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def fit_chat_prompt(
    message: str,
    budget: int,
    token_count: Callable[[str], int],
) -> str:
    """
    Trim unicode characters from the end of message until it fits within budget.
    Matches Rust fit_chat_prompt in yumon_rss.rs.
    """
    prompt = message.strip()
    while token_count(prompt) > budget:
        prompt = prompt[:-1]
    if not prompt.strip():
        raise ValueError("Checkpoint context cannot fit your message")
    return prompt


def language_prompt(
    message: str,
    memories: Iterable[Tuple[str, str]],
    tokenizer: Tokenizer,
    max_input_tokens: int,
) -> Optional[str]:
    """
    Formats a conversation prompt with recent memories that fit within the token budget.
    Memories are checked most-recent-first (reversed), then chronological in the prompt.
    An empty history keeps the original input. Matches language_prompt in Rust samples.rs.
    """
    token_count = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
    if token_count(message) > max_input_tokens:
        return None

    prompt = message
    turns: List[str] = []
    for human, yumon in reversed(list(memories)):
        turns.insert(0, f"Human: {human}\nYumon: {yumon}\n")
        candidate = f"{''.join(turns)}Human: {message}"
        if token_count(candidate) > max_input_tokens:
            break
        prompt = candidate

    return prompt


def extract_memories(history: Sequence[Any]) -> List[Tuple[str, str]]:
    """
    Extract memories from adjacent successful exchanges (user followed by yumon).
    Error messages or pending messages never form a pair.
    Matches history.windows(2) filtering in yumon_rss.rs.
    """
    memories: List[Tuple[str, str]] = []
    for i in range(len(history) - 1):
        m1, m2 = history[i], history[i + 1]
        r1 = m1.get("role") if isinstance(m1, dict) else getattr(m1, "role", "")
        t1 = m1.get("text") if isinstance(m1, dict) else getattr(m1, "text", "")
        r2 = m2.get("role") if isinstance(m2, dict) else getattr(m2, "role", "")
        t2 = m2.get("text") if isinstance(m2, dict) else getattr(m2, "text", "")
        if r1 == "user" and r2 == "yumon":
            memories.append((str(t1), str(t2)))
    return memories


def chat_prompt(
    message: str,
    history: Sequence[Any],
    context: int,
    tokenizer: Tokenizer,
) -> str:
    """
    Build a chat prompt with memory context matching yumon_rss.rs:
    - Reserve reply tokens (clamp(context // 4, 8, 64))
    - Calculate budget taking BOS and separator token into account
    - Trim message to budget
    - Incorporate chronological memories that fit within the budget
    """
    reply_tokens = max(8, min(64, context // 4))
    space_ids = tokenizer.encode(" ", add_special_tokens=False).ids
    sep_len = len(space_ids) if space_ids else 1
    overhead = 1 + sep_len + reply_tokens
    budget = context - overhead
    if budget <= 0:
        raise ValueError("Checkpoint context is too short")

    token_count = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
    fitted_message = fit_chat_prompt(message, budget, token_count)
    memories = extract_memories(history)
    prompt = language_prompt(fitted_message, memories, tokenizer, budget)
    if prompt is None:
        raise ValueError("Checkpoint context cannot fit your message")
    return prompt


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
    history: Optional[Sequence[Any]] = None,
    context: Optional[int] = None,
    debug: bool = False,
) -> str:
    ctx = context if context is not None else getattr(model.config, "max_seq_len", 256)
    if history is not None:
        prompt_text = chat_prompt(prompt, history, ctx, tokenizer)
    else:
        prompt_text = prompt

    encoded = tokenizer.encode(prompt_text, add_special_tokens=False).ids

    # Training format: [BOS_TOKEN] + prompt_ids + [SEP_TOKEN (space)] + reply_ids + [EOS_TOKEN]
    space_ids = tokenizer.encode(" ", add_special_tokens=False).ids
    sep_tokens = space_ids if space_ids else [146]
    prompt_ids = [BOS_TOKEN] + encoded + sep_tokens
    prompt_len = len(prompt_ids)

    if debug:
        print("\n--- [Prompt Sent to Model] ---")
        print(prompt_text)
        print(f"Token count: {prompt_len} / {ctx} (reserved: {max_new_tokens})")
        print("------------------------------\n")

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
    default_tok = "tokenizer.json" if os.path.exists("tokenizer.json") else "../yumon_bpe/tokenizer.json"
    parser = argparse.ArgumentParser(description="Generate replies with trained Yumon Pet Sparse MoE Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default=default_tok, help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string (if omitted, interactive mode starts)")
    parser.add_argument("--history", type=str, default=None, help="JSON history file path to load/save chat messages")
    parser.add_argument("--memory", nargs=2, action="append", metavar=("HUMAN", "YUMON"), help="Memory exchange pairs: --memory <human> <yumon>")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff")
    parser.add_argument("--debug", action="store_true", help="Print debug information including formatted prompt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)

    history: List[ChatMessage] = []
    if args.history:
        history = load_history(args.history)
        if history:
            print(f"Loaded {len(history)} messages from {args.history}")

    if args.memory:
        for human_msg, yumon_msg in args.memory:
            history.append(ChatMessage(role="user", text=human_msg))
            history.append(ChatMessage(role="yumon", text=yumon_msg))

    if args.prompt:
        history.append(ChatMessage(role="user", text=args.prompt))
        reply = generate_reply(
            model, tokenizer, args.prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            history=history,
            context=config.max_seq_len,
            debug=args.debug,
        )
        history.append(ChatMessage(role="yumon", text=reply))
        if len(history) > HISTORY_LIMIT:
            history = history[-HISTORY_LIMIT:]
        if args.history:
            save_history(args.history, history)

        print("\n=== Generation ===")
        print(f"Prompt: {args.prompt}")
        print(f"Reply:  {reply}")
    else:
        print("\nEntering interactive chat mode with Yumon Pet (type 'exit' or 'quit' to quit, 'clear' to reset, 'history' to view):")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input or user_input.lower() in ["exit", "quit"]:
                    break
                if user_input.lower() == "clear":
                    history.clear()
                    print("Conversation history cleared.")
                    continue
                if user_input.lower() == "history":
                    memories = extract_memories(history)
                    if not memories:
                        print("No memories stored yet.")
                    else:
                        print(f"\nStored memories ({len(memories)} pairs):")
                        for idx, (h, y) in enumerate(memories, 1):
                            print(f"  [{idx}] Human: {h}")
                            print(f"      Yumon: {y}")
                    continue

                history.append(ChatMessage(role="user", text=user_input))
                try:
                    reply = generate_reply(
                        model, tokenizer, user_input, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        history=history,
                        context=config.max_seq_len,
                        debug=args.debug,
                    )
                    history.append(ChatMessage(role="yumon", text=reply))
                    print(f"Yumon: {reply}")
                except Exception as err:
                    history.append(ChatMessage(role="error", text=f"Could not reply: {err}"))
                    print(f"Error: {err}")

                if len(history) > HISTORY_LIMIT:
                    history = history[-HISTORY_LIMIT:]
                if args.history:
                    save_history(args.history, history)

            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
