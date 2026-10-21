import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from .data import Images, inspect
from .paths import DATA_DIR, RESULTS_DIR
from .model import Encoder
from .clustering import cluster
from .losses import centroids, objective, update_memories
from .evaluate import extract, evaluate


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


class PKSampler(Sampler):
    def __init__(self, labels, p, k, steps, seed):
        self.groups = {
            int(c): np.flatnonzero(labels == c) for c in np.unique(labels) if c >= 0
        }
        if len(self.groups) < 2:
            raise RuntimeError(
                "Clustering produced fewer than two identities. Tune eps on validation; do not replace labels with ground truth."
            )
        self.p = min(p, len(self.groups))
        self.k = k
        self.steps = steps
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps):
            selected = rng.choice(list(self.groups), self.p, replace=False)
            yield [
                int(i)
                for c in selected
                for i in rng.choice(
                    self.groups[c], self.k, replace=len(self.groups[c]) < self.k
                )
            ]

    def __len__(self):
        return self.steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["inspect", "train", "evaluate"])
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--output", default=str(RESULTS_DIR / "full"))
    p.add_argument(
        "--variant", choices=["full", "baseline", "no_mlcor"], default="full"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--p", type=int, default=4)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eps", type=float, default=0.6)
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--checkpoint")
    p.add_argument("--split", choices=["validation", "testing"], default="validation")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "inspect":
        data = inspect(args.data)
        save_json(out / "dataset_audit.json", data)
        print(json.dumps(data, indent=2))
        return
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    if args.command == "evaluate" and not args.checkpoint:
        p.error("evaluate requires --checkpoint")
    model = Encoder(pretrained=not args.checkpoint).to(device)
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
        )
    config_name = (
        "config.json"
        if args.command == "train"
        else args.split + "_evaluation_config.json"
    )
    save_json(out / config_name, dict(vars(args), resolved_device=device))
    print("device:", device, flush=True)
    if args.command == "evaluate":
        results, examples = evaluate(
            model, Path(args.data) / (args.split + ".h5"), device, args.batch_size
        )
        save_json(out / (args.split + "_metrics.json"), results)
        save_json(out / (args.split + "_examples.json"), examples)
        print(json.dumps(results, indent=2), flush=True)
        return
    training = Images(Path(args.data) / "training.h5", train=True)
    clean = Images(Path(args.data) / "training.h5")
    optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    history = []
    best = -1
    for epoch in range(args.epochs):
        start = time.time()
        features = extract(model, clean, device, args.batch_size)
        print(f"epoch {epoch+1}: extracted {len(clean)} features", flush=True)
        labels, stats = cluster(
            [x.numpy() for x in features],
            eps=args.eps,
            k=args.neighbors,
            recover=args.variant == "full",
        )
        print("clustering:", stats, flush=True)
        np.save(out / f"labels_{epoch+1:03d}.npy", labels)
        save_json(out / f"clustering_{epoch+1:03d}.json", stats)
        sampler = PKSampler(labels, args.p, args.k, args.steps, args.seed + epoch)
        target = torch.as_tensor(labels, dtype=torch.long, device=device)
        memories = [x.to(device) for x in features]
        centers = [centroids(x, target) for x in memories]
        model.train()
        sums = {}
        for step, (images, ids) in enumerate(
            DataLoader(training, batch_sampler=sampler, num_workers=0)
        ):
            ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            current = model(images.to(device))
            loss, details = objective(
                current,
                memories,
                centers,
                target,
                ids,
                cem=args.variant in ("full", "no_mlcor"),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            update_memories(current, memories, centers, target, ids)
            for key, value in details.items():
                sums[key] = sums.get(key, 0) + value
            if (step + 1) % 10 == 0:
                print(
                    f"epoch {epoch+1} step {step+1}/{args.steps} loss {float(loss.detach()):.4f}",
                    flush=True,
                )
        scheduler.step()
        metrics, _ = evaluate(
            model, Path(args.data) / "validation.h5", device, args.batch_size
        )
        record = {
            "epoch": epoch + 1,
            "seconds": time.time() - start,
            "clustering": stats,
            "loss": {k: v / args.steps for k, v in sums.items()},
            "validation": metrics,
        }
        history.append(record)
        save_json(out / "history.json", history)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
        }
        torch.save(checkpoint, out / "last.pt")
        score = (metrics["original"]["mAP"] + metrics["low_resolution"]["mAP"]) / 2
        if score > best:
            best = score
            torch.save(checkpoint, out / "best.pt")
            save_json(out / "validation_metrics.json", metrics)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
