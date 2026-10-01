import torch
import time
from peft import LoraConfig, get_peft_model

def train(model, tokenizer, train_texts, val_texts):
    """Fine-tune `model` on `train_texts` (TinyStories samples).

    Any method works — full fine-tuning, LoRA via `peft`, prefix tuning,
    layer freezing, etc. The validator measures val_loss after your
    training and reports it — you don't need to print anything.

    Args:
        model:        AutoModelForCausalLM (distilgpt2), cuda + fp16.
        tokenizer:    matching AutoTokenizer.
        train_texts:  list[str]  -- TinyStories train samples
        val_texts:    list[str]  -- TinyStories val samples

    Returns:
        The trained model.
    """
    # Your code here.
    started = time.perf_counter()
    torch.manual_seed(42)

    if not train_texts:
        return model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = 'right'
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
        bias="none",
    )
    model = get_peft_model(model, config)

    parameters = [p for p in model.parameters() if p.requires_grad]
    for p in parameters:
        p.data = p.data.float()
    
    device = next(model.parameters()).device

    encoded = tokenizer(
        [text + tokenizer.eos_token for text in train_texts],
        truncation=True,
        max_length=256,
        padding=False,
    )

    samples = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(
            encoded["input_ids"], encoded["attention_mask"]
        )
    ]

    optimizer = torch.optim.AdamW(
        parameters, lr=5e-4, weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda")

    model.train()

    for epoch in range(3):
        order = torch.randperm(len(samples)).tolist()

        for start in range(0, len(order), 4):
            if time.perf_counter() - started >= 75:
                optimizer.zero_grad(set_to_none=True)
                return model.eval()

            batch = tokenizer.pad(
                [samples[i] for i in order[start:start + 4]],
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            batch = {name: value.to(device)
                     for name, value in batch.items()}

            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch, labels=labels).loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()

    optimizer.zero_grad(set_to_none=True)
    return model.eval()


