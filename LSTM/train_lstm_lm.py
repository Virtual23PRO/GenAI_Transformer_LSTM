import os, math, gzip, argparse, random, json
from pathlib import Path
from typing import List, Optional
import csv, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
SCALER = torch.cuda.amp.GradScaler()
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
TOKENIZER_ID = "speakleash/Bielik-7B-v0.1"

def _sample_next(logits, temperature=0.9, top_p=0.9):
    if temperature > 0:
        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        mask = cdf > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_id = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, next_id)
    else:
        return logits.argmax(dim=-1, keepdim=True)

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=150, temperature=0.9, top_p=0.9):
    model.eval()
    ids = tok_encode(tokenizer, prompt)
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device, dtype=torch.long)
    max_ctx = getattr(model, "max_len", 1024)
    for _ in range(max_new_tokens):
        inp = x[:, -max_ctx:]
        logp = model(inp)                         
        next_id = _sample_next(logp[:, -1, :].float(), temperature, top_p)
        x = torch.cat([x, next_id], dim=1)
    return tok_decode(tokenizer, x[0].tolist())

def tok_encode(tokenizer, text: str):
    try:
        out = tokenizer.encode(text, add_special_tokens=False)  
    except TypeError:
        out = tokenizer.encode(text)                            
    return out.ids if hasattr(out, "ids") else out            

def tok_decode(tokenizer, ids):
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(ids)

def load_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)

    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})

    return tok, tok.pad_token_id, tok.bos_token_id, tok.eos_token_id


def _read_text(fp: Path) -> str:
    if str(fp).endswith(".gz"):
        with gzip.open(fp, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return fp.read_text(encoding="utf-8", errors="ignore")

def encode_corpus(tokenizer, files, bos_id=None, eos_id=None):
    ids = []
    for fp in files:
        txt = _read_text(fp)
        for ln in txt.splitlines():
            if not ln.strip(): continue
            piece = tok_encode(tokenizer, ln)
            if bos_id is not None: piece = [bos_id] + piece
            if eos_id is not None: piece = piece + [eos_id]
            ids.extend(piece)
    return ids

class CLMDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len=128, stride: Optional[int] = None):
        self.ids = token_ids
        self.seq_len = seq_len
        self.stride = stride or seq_len  
        if len(self.ids) < (self.seq_len + 1):
            self.n = 0
        else:
            avail = len(self.ids) - (self.seq_len + 1)
            self.n = avail // self.stride + 1

    def __len__(self): return self.n

    def __getitem__(self, i):
        start = i * self.stride
        end = start + self.seq_len + 1
        chunk = self.ids[start:end]
        if len(chunk) < self.seq_len + 1:
            start = max(0, len(self.ids) - (self.seq_len + 1))
            chunk = self.ids[start:start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return {"input_ids": x, "labels": y}

def collate(batch):
    x = torch.stack([b["input_ids"] for b in batch])
    y = torch.stack([b["labels"]    for b in batch])
    return {"input_ids": x, "labels": y}

def log_metrics_csv(path: Path, row: dict, field_order):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if new:
            w.writeheader()
        w.writerow(row)


class LSTMLanguageModel(nn.Module):
    """
    Autoregresyjny LM na LSTM:
      - embedding -> LSTM (n warstw) -> dropout -> linear -> log_softmax
      - weight tying (lm.weight = tok.weight) opcjonalne (tu: domyślnie True)
    """
    def __init__(self, vocab_size, d_model=512, n_layers=2, dropout=0.2,
                 pad_id=None, tie_weights=True, max_len=4096):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len

        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.lm = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm.weight = self.tok.weight 

    def forward(self, x):
        # x: [B, L] (int)
        emb = self.drop(self.tok(x))          
        out, _ = self.lstm(emb)             
        out = self.drop(out)
        logits = self.lm(out)                
        return torch.log_softmax(logits, dim=-1)

def train_epoch(model, loader, opt, pad_id, amp_dtype=AMP_DTYPE, accum_steps=1, update_every=20):
    model.train()
    total_loss, total_tok = 0.0, 0
    opt.zero_grad(set_to_none=True)
    bar = tqdm(total=len(loader), desc="train", dynamic_ncols=True)

    for i, batch in enumerate(loader, 1):
        x = batch["input_ids"].to(next(model.parameters()).device, non_blocking=True)
        y = batch["labels"].to(x.device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            logp = model(x)
            loss = F.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1),
                              ignore_index=pad_id if pad_id is not None else -100)
            loss = loss / accum_steps

        SCALER.scale(loss).backward()
        if i % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            SCALER.step(opt); SCALER.update()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            n_tok = y.numel() if pad_id is None else (y != pad_id).sum().item()
            total_loss += (loss.item() * accum_steps) * n_tok
            total_tok  += n_tok

        if i % update_every == 0:
            ppl = math.exp(total_loss / max(1, total_tok))
            bar.set_postfix_str(f"ppl~{ppl:.1f}")
        bar.update(1)

    bar.close()
    return math.exp(total_loss / max(1, total_tok))

@torch.no_grad()
def eval_ppl(model, loader, pad_id):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for batch in tqdm(loader, desc="valid", leave=False, dynamic_ncols=True):
        x = batch["input_ids"].to(next(model.parameters()).device)
        y = batch["labels"].to(x.device)
        logp = model(x)
        loss = F.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1),
                          ignore_index=pad_id if pad_id is not None else -100, reduction="sum")
        total_loss += loss.item()
        total_tok  += y.numel() if pad_id is None else (y != pad_id).sum().item()
    return math.exp(total_loss / max(1, total_tok))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="out_lstm")

    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--stride",  type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--d_model", type=int, default=512)    
    ap.add_argument("--n_layers", type=int, default=2)     
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--gen", action="store_true", help="Tylko generacja (bez treningu)")
    ap.add_argument("--ckpt", type=str, default=None, help="Ścieżka do .pt (last/best)")
    ap.add_argument("--prompts", nargs="*", default=[], help="Lista promptów do wygenerowania")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.02, help="Udział walidacji (0–1) przy dzieleniu z jednego strumienia")

    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer, pad_id, bos_id, eos_id = load_tokenizer(args)

    vocab_size = tokenizer.vocab_size

    vocab_size = max(
        vocab_size,
        (pad_id or -1) + 1,
        (bos_id or -1) + 1,
        (eos_id or -1) + 1,
    )

    if not args.gen:
        out_tok = Path(args.output_dir)
        if not (out_tok / "tokenizer_hf").exists():
            (out_tok / "tokenizer_hf").mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(out_tok / "tokenizer_hf"))

    cfg_path = Path(args.output_dir) / "model_config.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.gen:
        assert cfg_path.exists(), f"Brak {cfg_path}. Najpierw odpal trening, żeby zapisać konfigurację."
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        d_model  = cfg.get("d_model", args.d_model)
        n_layers = cfg.get("n_layers", args.n_layers)
        dropout  = cfg.get("dropout",  args.dropout)
        pad_id   = cfg.get("pad_id",   pad_id)
        max_len  = cfg.get("max_len",  4096)

        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            pad_id=pad_id,
            tie_weights=True,
            max_len=max_len
        ).to(device)

        assert args.ckpt is not None, "Podaj --ckpt ze ścieżką do wag (np. out_lstm/last.pt)"
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=True)

        prompts = args.prompts or ["To jest początek zdania."]
        for pr in prompts:
            out = generate_text(
                model, tokenizer, pr,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print("\nPROMPT:", pr)
            print("OUTPUT:", out)
        return

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pad_id=pad_id,
        tie_weights=True,
        max_len=4096
    ).to(device)

    cfg = {
        "model_type": "lstm",
        "d_model":  args.d_model,
        "n_layers": args.n_layers,
        "dropout":  args.dropout,
        "pad_id":   pad_id,
        "max_len":  4096,
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    all_files = sorted(Path(args.data_dir).glob("*.txt")) + sorted(Path(args.data_dir).glob("*.txt.gz"))
    assert all_files, "Brak plików .txt/.txt.gz w data_dir"

    all_ids = encode_corpus(tokenizer, all_files, bos_id, eos_id)
    val_ratio = getattr(args, "val_ratio", 0.02)
    cut = int((1.0 - val_ratio) * len(all_ids))
    train_ids, valid_ids = all_ids[:cut], all_ids[cut:]

    train_ds = CLMDataset(train_ids, seq_len=args.seq_len, stride=args.stride or args.seq_len)
    valid_ds = CLMDataset(valid_ids, seq_len=args.seq_len, stride=args.stride or args.seq_len)

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          collate_fn=collate, num_workers=args.workers, pin_memory=True,
                          prefetch_factor=4, persistent_workers=True)
    dl_valid = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                          collate_fn=collate, num_workers=max(1, args.workers//2), pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)



    log_path   = Path(args.output_dir) / "train_log.csv"
    log_fields = ["epoch", "split", "loss", "ppl", "epoch_time_s", "elapsed_s", "tokens_per_s"]
    run_start  = time.time()

    best = float("inf")
    for e in range(1, args.epochs + 1):
        epoch_start = time.time()
        tr_ppl = train_epoch(model, dl_train, opt, pad_id, accum_steps=args.accum_steps)
        va_ppl = eval_ppl(model, dl_valid, pad_id) if len(valid_ds) > 0 else float("nan")


        epoch_sec   = time.time() - epoch_start
        elapsed_sec = time.time() - run_start
        steps = len(dl_train)
        tokens_epoch_est = steps * args.batch_size * args.seq_len
        tps = tokens_epoch_est / max(1e-9, epoch_sec)
    
        tr_loss = math.log(tr_ppl) if tr_ppl > 0 else float("nan")
        va_loss = math.log(va_ppl) if (va_ppl > 0 and not math.isnan(va_ppl)) else float("nan")
    
        log_metrics_csv(log_path, {
            "epoch": e,
            "split": "train",
            "loss": tr_loss,
            "ppl": tr_ppl,
            "epoch_time_s": round(epoch_sec, 3),
            "elapsed_s": round(elapsed_sec, 3),
            "tokens_per_s": round(tps, 1),
        }, field_order=log_fields)
    
        if not math.isnan(va_ppl):
            log_metrics_csv(log_path, {
                "epoch": e,
                "split": "valid",
                "loss": va_loss,
                "ppl": va_ppl,
                "epoch_time_s": round(epoch_sec, 3),
                "elapsed_s": round(elapsed_sec, 3),
                "tokens_per_s": round(tps, 1),
            }, field_order=log_fields)



        print(f"[epoch {e}] train PPL={tr_ppl:.2f} | valid PPL={va_ppl:.2f}")
        torch.save(model.state_dict(), Path(args.output_dir) / "last.pt")
        if va_ppl < best:
            best = va_ppl
            torch.save(model.state_dict(), Path(args.output_dir) / "best.pt")

    print("Done.")


if __name__ == "__main__":
    main()
