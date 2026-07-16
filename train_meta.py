"""BCI IV 2a 跨被试二阶 MAML 训练入口。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

from meta_learning import (
    FoldResult,
    MetaTrainConfig,
    select_device,
    set_random_seed,
    train_loso_fold,
)
from moabb_data import DEFAULT_MOABB_DATA_ROOT, load_bciciv2a
from models.MetaEEGNet import MetaEEGNet


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "22通道Transformer Inner loop + 3通道卷积Outer loop的二阶MAML"))

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_MOABB_DATA_ROOT,
        help="MOABB下载目录，默认位于项目的data/moabb。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="检查点和结果目录。")
    parser.add_argument(
        "--test-subject",
        type=int,
        default=0,
        help="1～9表示单个LOSO折，0表示依次运行全部9折。")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=32.0)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--meta-epochs", type=int, default=20)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--meta-lr", type=float, default=0.001)
    parser.add_argument("--support-batch-per-subject", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-lr", type=float, default=0.0005)
    parser.add_argument("--finetune-batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--f1", type=int, default=8)
    parser.add_argument("--f2", type=int, default=16)
    parser.add_argument("--kernel-length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--transformer-heads", type=int, default=2)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-ratio", type=int, default=4)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.test_subject not in range(0, 10):
        raise ValueError("--test-subject必须为0～9。")
    positive_integer_names = (
        "meta_epochs", "inner_steps", "support_batch_per_subject",
        "query_batch_size", "finetune_epochs", "finetune_batch_size",
        "f1", "f2", "kernel_length", "transformer_heads",
        "transformer_layers", "transformer_ff_ratio",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')}必须大于0。")
    if args.fmin >= args.fmax:
        raise ValueError("fmin必须小于fmax。")


def _make_train_config(args: argparse.Namespace) -> MetaTrainConfig:
    return MetaTrainConfig(
        meta_epochs=args.meta_epochs,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        support_batch_per_subject=args.support_batch_per_subject,
        query_batch_size=args.query_batch_size,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_batch_size=args.finetune_batch_size,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed)


def _make_model(args: argparse.Namespace, samples: int) -> MetaEEGNet:
    return MetaEEGNet(
        n_classes=4,
        samples=samples,
        f1=args.f1,
        f2=args.f2,
        kernel_length=args.kernel_length,
        dropout_rate=args.dropout,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_ratio=args.transformer_ff_ratio)


def _write_results(
        output_dir: Path,
        args: argparse.Namespace,
        config: MetaTrainConfig,
        results: List[FoldResult]) -> Path:
    accuracies = np.asarray([result.accuracy for result in results])
    kappas = np.asarray([result.kappa for result in results])
    payload = {
        "data_root": str(args.data_root.expanduser().resolve()),
        "preprocessing": {"fmin": args.fmin, "fmax": args.fmax},
        "meta_config": asdict(config),
        "model_config": {
            "f1": args.f1,
            "f2": args.f2,
            "kernel_length": args.kernel_length,
            "dropout": args.dropout,
            "transformer_heads": args.transformer_heads,
            "transformer_layers": args.transformer_layers,
            "transformer_ff_ratio": args.transformer_ff_ratio,
        },
        "folds": [asdict(result) for result in results],
        "summary": {
            "accuracy_mean": float(accuracies.mean()),
            "accuracy_std": float(accuracies.std(ddof=0)),
            "kappa_mean": float(kappas.mean()),
            "kappa_std": float(kappas.std(ddof=0)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "loso_results.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return result_path


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    _validate_arguments(args)

    subject_data, data_root = load_bciciv2a(
        data_root=args.data_root,
        subjects=range(1, 10),
        fmin=args.fmin,
        fmax=args.fmax)
    first_subject = subject_data[min(subject_data)]
    print(
        f"MOABB数据目录：{data_root}\n"
        f"被试数：{len(subject_data)}，单被试形状：{first_subject.x.shape}")
    if args.download_only:
        return

    config = _make_train_config(args)
    device = select_device(args.device)
    print(f"训练设备：{device}")

    test_subjects = (
        list(range(1, 10)) if args.test_subject == 0
        else [args.test_subject]
    )
    results: List[FoldResult] = []
    for test_subject in test_subjects:
        # 每个LOSO折从独立且可复现的模型初始化开始。
        set_random_seed(args.seed + test_subject)
        model = _make_model(args, samples=first_subject.x.shape[-1])
        result, _ = train_loso_fold(
            subject_data=subject_data,
            test_subject=test_subject,
            model=model,
            config=config,
            device=device,
            output_dir=args.output_dir)
        results.append(result)
        print(
            f"test_subject={test_subject} "
            f"accuracy={result.accuracy:.4f} kappa={result.kappa:.4f}")

    result_path = _write_results(args.output_dir, args, config, results)
    print(f"结果已保存：{result_path.resolve()}")


if __name__ == "__main__":
    main()
