"""跨被试二阶 MAML 的训练与测试流程。"""

import copy
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from moabb_data import SubjectTrials, THREE_CHANNEL_INDICES, split_subject_sessions
from models.MetaEEGNet import CONV_BRANCH, TRANSFORMER_BRANCH, MetaEEGNet


@dataclass(frozen=True)
class MetaTrainConfig:
    """单个 LOSO 折的训练配置。"""

    meta_epochs: int = 20
    inner_steps: int = 1
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    support_batch_per_subject: int = 8
    query_batch_size: int = 64
    finetune_epochs: int = 10
    finetune_lr: float = 0.0005
    finetune_batch_size: int = 64
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    seed: int = 42


@dataclass(frozen=True)
class FoldResult:
    test_subject: int
    accuracy: float
    kappa: float
    finetune_session: str
    test_session: str
    checkpoint: str


class SubjectBatchPool:
    """在内存中循环遍历一个被试的全部试次。"""

    def __init__(self, trials: SubjectTrials, seed: int):
        self.x = torch.from_numpy(trials.x)
        self.y = torch.from_numpy(trials.y)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.y))
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.y)

    def _reshuffle(self) -> None:
        self.order = self.rng.permutation(len(self.y))
        self.cursor = 0

    def next_indices(self, batch_size: int) -> np.ndarray:
        """循环取样；跨越末尾时重新打乱，保证长期覆盖全部试次。"""
        if batch_size < 1:
            raise ValueError("batch_size必须大于0。")

        pieces = []
        remaining = batch_size
        while remaining > 0:
            available = len(self.order) - self.cursor
            take = min(remaining, available)
            pieces.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor == len(self.order):
                self._reshuffle()
        return np.concatenate(pieces)

    def iter_epoch_indices(self, batch_size: int) -> Iterator[np.ndarray]:
        """将全部试次随机遍历一次，不重复、不遗漏。"""
        order = self.rng.permutation(len(self.y))
        for start in range(0, len(order), batch_size):
            yield order[start:start + batch_size]

    def make_batch(self,
                   indices: Sequence[int],
                   use_three_channels: bool,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        x = self.x.index_select(0, tensor_indices)
        if use_three_channels:
            channel_indices = torch.as_tensor(
                THREE_CHANNEL_INDICES, dtype=torch.long)
            x = x.index_select(1, channel_indices)

        # EEGNet 的输入格式为 [批次, 1, 通道, 时间点]。
        x = x.unsqueeze(1).to(device=device, dtype=torch.float32)
        y = self.y.index_select(0, tensor_indices).to(
            device=device, dtype=torch.long)
        return x, y


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """根据参数与硬件情况选择 CUDA、MPS 或 CPU。"""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _support_batch(
        pools: Dict[int, SubjectBatchPool],
        support_subjects: Sequence[int],
        batch_per_subject: int,
        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """从每个 Support 被试等量取样，避免样本数主导损失。"""
    x_batches = []
    y_batches = []
    for subject in support_subjects:
        pool = pools[subject]
        indices = pool.next_indices(batch_per_subject)
        x, y = pool.make_batch(
            indices, use_three_channels=False, device=device)
        x_batches.append(x)
        y_batches.append(y)
    return torch.cat(x_batches, dim=0), torch.cat(y_batches, dim=0)


def second_order_inner_update(
        model: MetaEEGNet,
        initial_parameters: OrderedDict,
        pools: Dict[int, SubjectBatchPool],
        support_subjects: Sequence[int],
        config: MetaTrainConfig,
        device: torch.device,
        criterion: nn.Module) -> OrderedDict:
    """使用22通道 Transformer 分支生成可二阶求导的 fast weights。"""
    fast_parameters = OrderedDict(initial_parameters)
    adapt_names = model.inner_parameter_names()

    for _ in range(config.inner_steps):
        support_x, support_y = _support_batch(
            pools=pools,
            support_subjects=support_subjects,
            batch_per_subject=config.support_batch_per_subject,
            device=device)
        # Flash / Efficient Attention 的部分后端没有实现二阶导数；
        # MAML 必须使用数学型 SDPA 内核保留完整的元梯度链。
        with sdpa_kernel(SDPBackend.MATH):
            logits = functional_call(
                model,
                fast_parameters,
                (support_x, TRANSFORMER_BRANCH))
        inner_loss = criterion(logits, support_y)

        adapted_tensors = tuple(fast_parameters[name] for name in adapt_names)
        gradients = torch.autograd.grad(
            inner_loss,
            adapted_tensors,
            create_graph=True)

        updated_parameters = OrderedDict(fast_parameters)
        for name, gradient in zip(adapt_names, gradients):
            updated_parameters[name] = (
                fast_parameters[name] - config.inner_lr * gradient)
        fast_parameters = updated_parameters

    return fast_parameters


def _compute_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     n_classes: int = 4) -> Tuple[float, float]:
    """计算准确率和 Cohen's Kappa，避免对训练流程增加额外状态。"""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("指标输入为空或长度不一致。")

    accuracy = float(np.mean(y_true == y_pred))
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for target, prediction in zip(y_true, y_pred):
        confusion[int(target), int(prediction)] += 1

    total = confusion.sum()
    observed = np.trace(confusion) / total
    expected = (
        confusion.sum(axis=1) @ confusion.sum(axis=0)) / (total * total)
    kappa = 0.0 if math.isclose(1.0 - expected, 0.0) else (
        observed - expected) / (1.0 - expected)
    return accuracy, float(kappa)


def _finetune_conv_branch(
        model: MetaEEGNet,
        trials: SubjectTrials,
        config: MetaTrainConfig,
        device: torch.device) -> MetaEEGNet:
    """使用测试被试第一个会话的三通道数据微调卷积分支。"""
    finetuned_model = copy.deepcopy(model).to(device)
    allowed_names = set(finetuned_model.conv_finetune_parameter_names())
    trainable_parameters = []
    for name, parameter in finetuned_model.named_parameters():
        parameter.requires_grad_(name in allowed_names)
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config.finetune_lr,
        weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    pool = SubjectBatchPool(trials, seed=config.seed + 1000)

    finetuned_model.train()
    for _ in range(config.finetune_epochs):
        for indices in pool.iter_epoch_indices(config.finetune_batch_size):
            x, y = pool.make_batch(
                indices, use_three_channels=True, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(finetuned_model(x, CONV_BRANCH), y)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable_parameters, config.grad_clip)
            optimizer.step()
    return finetuned_model


@torch.no_grad()
def _evaluate_conv_branch(
        model: MetaEEGNet,
        trials: SubjectTrials,
        batch_size: int,
        device: torch.device) -> Tuple[float, float]:
    """在测试被试第二个会话上评估三通道卷积分支。"""
    model.eval()
    pool = SubjectBatchPool(trials, seed=0)
    targets: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    for indices in pool.iter_epoch_indices(batch_size):
        x, y = pool.make_batch(
            indices, use_three_channels=True, device=device)
        logits = model(x, CONV_BRANCH)
        targets.append(y.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return _compute_metrics(
        np.concatenate(targets), np.concatenate(predictions))


def train_loso_fold(
        subject_data: Dict[int, SubjectTrials],
        test_subject: int,
        model: MetaEEGNet,
        config: MetaTrainConfig,
        device: torch.device,
        output_dir: Path) -> Tuple[FoldResult, List[dict]]:
    """训练一个留一被试折，并在被留出的被试上微调和测试。"""
    if test_subject not in subject_data:
        raise ValueError(f"缺少测试被试 {test_subject} 的数据。")

    set_random_seed(config.seed + test_subject)
    model = model.to(device)
    train_subjects = sorted(set(subject_data) - {test_subject})
    pools = {
        subject: SubjectBatchPool(
            subject_data[subject], seed=config.seed + subject)
        for subject in train_subjects
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.meta_lr,
        weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed + test_subject)
    history: List[dict] = []

    for epoch in range(1, config.meta_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_correct = 0
        epoch_count = 0

        # 每个训练被试轮流作为 Query；其全部试次在本轮恰好遍历一次。
        for query_subject in rng.permutation(train_subjects).tolist():
            support_subjects = [
                subject for subject in train_subjects
                if subject != query_subject
            ]
            query_pool = pools[query_subject]

            for query_indices in query_pool.iter_epoch_indices(
                    config.query_batch_size):
                initial_parameters = OrderedDict(model.named_parameters())
                fast_parameters = second_order_inner_update(
                    model=model,
                    initial_parameters=initial_parameters,
                    pools=pools,
                    support_subjects=support_subjects,
                    config=config,
                    device=device,
                    criterion=criterion)

                query_x, query_y = query_pool.make_batch(
                    query_indices,
                    use_three_channels=True,
                    device=device)
                query_logits = functional_call(
                    model,
                    fast_parameters,
                    (query_x, CONV_BRANCH))
                outer_loss = criterion(query_logits, query_y)

                optimizer.zero_grad(set_to_none=True)
                outer_loss.backward()
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip)
                optimizer.step()

                epoch_losses.append(float(outer_loss.detach().cpu()))
                epoch_correct += int(
                    (query_logits.argmax(dim=1) == query_y).sum().item())
                epoch_count += len(query_y)

        epoch_record = {
            "epoch": epoch,
            "outer_loss": float(np.mean(epoch_losses)),
            "outer_accuracy": epoch_correct / epoch_count,
        }
        history.append(epoch_record)
        print(
            f"fold={test_subject} epoch={epoch}/{config.meta_epochs} "
            f"loss={epoch_record['outer_loss']:.4f} "
            f"acc={epoch_record['outer_accuracy']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"loso_subject_{test_subject}.pt"
    finetune_trials, test_trials = split_subject_sessions(
        subject_data[test_subject])
    finetuned_model = _finetune_conv_branch(
        model=model,
        trials=finetune_trials,
        config=config,
        device=device)
    accuracy, kappa = _evaluate_conv_branch(
        model=finetuned_model,
        trials=test_trials,
        batch_size=config.finetune_batch_size,
        device=device)

    # 同时保存元训练状态和测试被试微调状态，便于复现实验结果。
    torch.save({
        "model_state_dict": model.state_dict(),
        "finetuned_model_state_dict": finetuned_model.state_dict(),
        "test_subject": test_subject,
        "finetune_session": str(np.unique(finetune_trials.sessions)[0]),
        "test_session": str(np.unique(test_trials.sessions)[0]),
        "accuracy": accuracy,
        "kappa": kappa,
        "config": asdict(config),
        "history": history,
    }, checkpoint_path)

    result = FoldResult(
        test_subject=test_subject,
        accuracy=accuracy,
        kappa=kappa,
        finetune_session=str(np.unique(finetune_trials.sessions)[0]),
        test_session=str(np.unique(test_trials.sessions)[0]),
        checkpoint=str(checkpoint_path))
    return result, history
