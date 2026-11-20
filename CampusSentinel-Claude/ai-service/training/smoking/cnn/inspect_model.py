"""Stage 1 check for SmokingCNN: summary, parameter count and dummy-input tests.

Uses RANDOM tensors only. It proves the architecture is wired correctly;
it says NOTHING about how well the model detects smoking (it is untrained).

Run from the ai-service folder:
    python -m training.smoking.cnn.inspect_model
    python -m training.smoking.cnn.inspect_model --memory-probe          (CUDA only)
    python -m training.smoking.cnn.inspect_model --memory-probe --amp    (mixed precision)

Exit code 0 = all checks passed, 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import platform
import sys
import traceback

import torch
from torch import nn

from training.common.reproducibility import get_device, set_seed
from training.smoking.cnn.model import INPUT_SIZE, SmokingCNN, count_parameters

# Hand-computed expectation for the default architecture (see README):
#   conv weights 3·3·(3·32 + 32·64 + 64·128 + 128·256) = 387,936
#   batchnorm    2·(32 + 64 + 128 + 256)              =     960
#   linear 1     256·256 + 256                        =  65,792
#   linear 2     256·1 + 1                            =     257
EXPECTED_PARAMS = 387_936 + 960 + 65_792 + 257  # = 454,945


# --------------------------------------------------------------------------- helpers
class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def run(self, name: str, fn) -> None:
        try:
            detail = fn() or ""
            self.results.append((name, True, detail))
            print(f"  [PASS] {name}{'  — ' + detail if detail else ''}")
        except Exception as exc:  # report and continue with remaining checks
            self.results.append((name, False, str(exc)))
            print(f"  [FAIL] {name}  — {exc}")
            traceback.print_exc(limit=1)

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)


def layer_table(model: nn.Module, device: torch.device) -> None:
    """Print output shape and parameter count of every leaf layer using forward hooks."""
    rows: list[tuple[str, str, tuple[int, ...], int]] = []
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            def hook(mod, _inp, out, name=name):
                params = sum(p.numel() for p in mod.parameters(recurse=False))
                rows.append((name, mod.__class__.__name__, tuple(out.shape), params))
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device))
    for h in hooks:
        h.remove()

    print(f"\n  {'Layer':<16}{'Type':<20}{'Output shape':<24}{'Params':>10}")
    print("  " + "-" * 70)
    for name, kind, shape, params in rows:
        print(f"  {name:<16}{kind:<20}{str(shape):<24}{params:>10,}")
    print("  " + "-" * 70)


def receptive_field(num_blocks: int = 4) -> int:
    """Receptive field of one final feature-map cell, in input pixels.
    Each block = conv3x3 (stride 1) then maxpool2 (stride 2)."""
    rf, jump = 1, 1
    for _ in range(num_blocks):
        rf += (3 - 1) * jump  # conv 3x3
        rf += (2 - 1) * jump  # maxpool 2x2
        jump *= 2
    return rf


def memory_probe(device: torch.device, amp: bool, batch_sizes: list[int]) -> None:
    """Measure peak GPU memory for one real training step (forward + backward + Adam)."""
    print(f"\n  Training-step memory on {torch.cuda.get_device_name(0)} (AMP={'on' if amp else 'off'})")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Total VRAM reported by PyTorch: {total_gb:.2f} GB\n")
    print(f"  {'Batch':>6}  {'Peak allocated':>15}  {'Peak reserved':>14}  Result")

    for bs in batch_sizes:
        model = SmokingCNN().to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(bs, 3, INPUT_SIZE, INPUT_SIZE, device=device)
            y = torch.randint(0, 2, (bs, 1), device=device).float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            alloc = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            print(f"  {bs:>6}  {alloc:>12.2f} GB  {reserved:>11.2f} GB  ok")
        except torch.cuda.OutOfMemoryError:
            print(f"  {bs:>6}  {'—':>15}  {'—':>14}  OUT OF MEMORY")
        finally:
            del model, opt
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and sanity-check SmokingCNN.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    parser.add_argument("--memory-probe", action="store_true", help="measure GPU memory per batch size")
    parser.add_argument("--amp", action="store_true", help="use mixed precision in the memory probe")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    args = parser.parse_args()

    # Windows: make symbols like × and → print safely even when output is redirected to a file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    print("=" * 74)
    print("SmokingCNN — Stage 1 architecture check (random inputs, UNTRAINED model)")
    print("=" * 74)
    print(f"  Python   {platform.python_version()}   PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}"
          + (f"  ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})" if torch.cuda.is_available() else ""))
    print(f"  Device used: {device}")

    model = SmokingCNN().to(device)
    print("\n" + str(model))
    layer_table(model, device)

    total, trainable = count_parameters(model)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"  Parameters: {total:,} total, {trainable:,} trainable  (~{size_mb:.2f} MB as float32)")
    print(f"  Receptive field of one final cell: {receptive_field()}×{receptive_field()} pixels "
          f"(of a {INPUT_SIZE}×{INPUT_SIZE} image)")

    print("\nChecks")
    checks = Checks()

    def params_match():
        assert total == EXPECTED_PARAMS, f"got {total:,}, expected {EXPECTED_PARAMS:,}"
        return f"{total:,} = hand-computed value"

    def output_shape():
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE, device=device))
        assert out.shape == (4, 1), f"got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "non-finite values in output"
        return f"(4, 3, 224, 224) → {tuple(out.shape)}"

    def no_sigmoid():
        found = [n for n, m in model.named_modules() if isinstance(m, nn.Sigmoid)]
        assert not found, f"Sigmoid modules found: {found}"
        return "forward() returns raw logits"

    def backward_pass():
        model.train()
        model.zero_grad(set_to_none=True)
        x = torch.randn(8, 3, INPUT_SIZE, INPUT_SIZE, device=device)
        y = torch.randint(0, 2, (8, 1), device=device).float()
        loss = nn.BCEWithLogitsLoss()(model(x), y)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not missing, f"no gradient for: {missing}"
        assert not bad, f"non-finite gradient for: {bad}"
        assert 0.5 < loss.item() < 0.9, f"initial loss {loss.item():.4f} far from ln(2)≈0.693"
        return f"initial loss on random data = {loss.item():.4f} (ln 2 ≈ 0.693 = 'no idea yet')"

    def eval_is_deterministic():
        model.eval()
        x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
        with torch.no_grad():
            a, b = model(x), model(x)
        assert torch.equal(a, b), "eval-mode outputs differ (dropout/BN not frozen?)"
        return "dropout off, BatchNorm uses running stats"

    def seed_reproducible():
        set_seed(args.seed)
        m1 = SmokingCNN()
        set_seed(args.seed)
        m2 = SmokingCNN()
        for (n, p1), (_, p2) in zip(m1.state_dict().items(), m2.state_dict().items()):
            assert torch.equal(p1, p2), f"parameter {n} differs"
        return f"same seed ({args.seed}) → identical initial weights"

    def rejects_bad_input():
        try:
            model(torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE, device=device))  # grayscale
        except ValueError:
            return "grayscale (1-channel) input raises ValueError"
        raise AssertionError("1-channel input was accepted")

    def predict_proba_range():
        p = model.predict_proba(torch.randn(3, 3, INPUT_SIZE, INPUT_SIZE, device=device))
        assert p.shape == (3,) and ((p >= 0) & (p <= 1)).all(), f"bad probabilities {p}"
        return "sigmoid applied only in predict_proba(), values in [0, 1]"

    checks.run("Parameter count", params_match)
    checks.run("Output shape", output_shape)
    checks.run("No sigmoid inside the model", no_sigmoid)
    checks.run("Backward pass with BCEWithLogitsLoss", backward_pass)
    checks.run("Eval mode is deterministic", eval_is_deterministic)
    checks.run("Seeded initialisation is reproducible", seed_reproducible)
    checks.run("Input validation", rejects_bad_input)
    checks.run("predict_proba()", predict_proba_range)

    if args.memory_probe:
        if device.type != "cuda":
            print("\n  --memory-probe skipped: CUDA not available (or --cpu given).")
        else:
            memory_probe(device, args.amp, args.batch_sizes)

    passed = sum(ok for _, ok, _ in checks.results)
    print(f"\nSummary: {passed}/{len(checks.results)} checks passed.")
    return 0 if checks.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
