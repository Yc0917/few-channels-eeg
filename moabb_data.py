"""使用 MOABB 加载 BCI Competition IV 2a。"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np


MOABB_BNCI_PATH_ENV = "MNE_DATASETS_BNCI_PATH"
DEFAULT_MOABB_DATA_ROOT = Path(__file__).resolve().parent / "data" / "moabb"

BCI_IV_2A_CHANNELS: Tuple[str, ...] = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
    "P2", "POz",
)
THREE_CHANNEL_NAMES: Tuple[str, ...] = ("C3", "Cz", "C4")
THREE_CHANNEL_INDICES: Tuple[int, ...] = tuple(
    BCI_IV_2A_CHANNELS.index(name) for name in THREE_CHANNEL_NAMES)

CLASS_NAMES: Tuple[str, ...] = (
    "left_hand", "right_hand", "feet", "tongue")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class SubjectTrials:
    """一个被试的全部试次及对应 MOABB 会话信息。"""

    x: np.ndarray
    y: np.ndarray
    sessions: np.ndarray
    runs: np.ndarray


def configure_moabb_data_root(
        data_root: Optional[Union[str, Path]] = None) -> Path:
    """为当前进程设置 MOABB 的 BNCI 数据下载目录。"""
    if data_root is None:
        data_root = os.environ.get(
            MOABB_BNCI_PATH_ENV, DEFAULT_MOABB_DATA_ROOT)

    resolved_root = Path(data_root).expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    # 使用环境变量避免改写用户全局的 MNE 配置文件。
    os.environ[MOABB_BNCI_PATH_ENV] = str(resolved_root)
    return resolved_root


def create_bciciv2a_dataset(
        data_root: Optional[Union[str, Path]] = None):
    """创建 MOABB 数据集对象，不主动触发整库下载。"""
    resolved_root = configure_moabb_data_root(data_root)
    try:
        from moabb.datasets import BNCI2014_001
    except ImportError as exc:
        raise ImportError(
            "缺少 MOABB，请先执行：pip install -r requirements.txt") from exc
    return BNCI2014_001(), resolved_root


def load_bciciv2a(
        data_root: Optional[Union[str, Path]] = None,
        subjects: Iterable[int] = range(1, 10),
        fmin: float = 8.0,
        fmax: float = 32.0) -> Tuple[Dict[int, SubjectTrials], Path]:
    """通过 MOABB 返回按被试组织的四分类运动想象试次。

    MOABB 返回的单个试次形状为 ``[22, 1001]``。本函数只转换为
    ``float32`` 和固定整数标签，不额外改变 MOABB 的预处理结果。
    """
    dataset, resolved_root = create_bciciv2a_dataset(data_root)
    try:
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise ImportError(
            "缺少 MOABB，请先执行：pip install -r requirements.txt") from exc

    subject_list = [int(subject) for subject in subjects]
    invalid_subjects = sorted(set(subject_list) - set(range(1, 10)))
    if invalid_subjects:
        raise ValueError(f"BCI IV 2a 被试编号必须为 1～9：{invalid_subjects}")

    paradigm = MotorImagery(n_classes=4, fmin=fmin, fmax=fmax)
    x, labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=subject_list)

    if x.ndim != 3 or x.shape[1] != len(BCI_IV_2A_CHANNELS):
        raise ValueError(
            f"MOABB 返回了非预期形状 {x.shape}，期望 [试次, 22, 时间点]。")

    unknown_labels = sorted(set(labels) - set(CLASS_TO_INDEX))
    if unknown_labels:
        raise ValueError(f"发现未知类别标签：{unknown_labels}")

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray([CLASS_TO_INDEX[label] for label in labels], dtype=np.int64)
    metadata_subjects = metadata["subject"].to_numpy(dtype=np.int64)
    metadata_sessions = metadata["session"].astype(str).to_numpy()
    metadata_runs = metadata["run"].astype(str).to_numpy()

    by_subject: Dict[int, SubjectTrials] = {}
    for subject in subject_list:
        mask = metadata_subjects == subject
        if not np.any(mask):
            raise ValueError(f"MOABB 未返回被试 {subject} 的数据。")
        by_subject[subject] = SubjectTrials(
            x=x[mask],
            y=y[mask],
            sessions=metadata_sessions[mask],
            runs=metadata_runs[mask])

    return by_subject, resolved_root


def select_three_channels(x: np.ndarray) -> np.ndarray:
    """从 ``[..., 22, time]`` 提取 C3、Cz、C4。"""
    if x.ndim < 2 or x.shape[-2] != len(BCI_IV_2A_CHANNELS):
        raise ValueError(f"输入形状 {x.shape} 的倒数第二维不是22通道。")
    return np.take(x, THREE_CHANNEL_INDICES, axis=-2)


def split_subject_sessions(
        trials: SubjectTrials) -> Tuple[SubjectTrials, SubjectTrials]:
    """按 MOABB 的两个会话划分微调集和最终测试集。"""
    session_names = sorted(np.unique(trials.sessions).tolist())
    if len(session_names) != 2:
        raise ValueError(f"期望两个会话，实际得到：{session_names}")

    splits = []
    for session_name in session_names:
        mask = trials.sessions == session_name
        splits.append(SubjectTrials(
            x=trials.x[mask],
            y=trials.y[mask],
            sessions=trials.sessions[mask],
            runs=trials.runs[mask]))
    return splits[0], splits[1]
