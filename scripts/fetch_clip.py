import argparse, json, hashlib, os, torch, open_clip

def sha256(path, buf=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32", help="e.g., ViT-B-32 or ViT-B-16")
    ap.add_argument("--pretrained", default="openai", help="openai or laion2b_s34b_b79k etc.")
    ap.add_argument("--outdir", default="models/clip")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save_dir = os.path.join(args.outdir, f"{args.model}-{args.pretrained}")
    os.makedirs(save_dir, exist_ok=True)

    # This will auto-download to cache if not present
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device="cpu"
    )
    ckpt_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)

    info = {
        "model": args.model,
        "pretrained": args.pretrained,
        "format": "torch.state_dict",
        "file": "model.pt",
        "sha256": sha256(ckpt_path),
        "license": "MIT (open-clip-torch); see THIRD_PARTY_NOTICES.md"
    }
    with open(os.path.join(save_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved: {ckpt_path}")
    print(f"SHA256: {info['sha256']}")

if __name__ == "__main__":
    main()
