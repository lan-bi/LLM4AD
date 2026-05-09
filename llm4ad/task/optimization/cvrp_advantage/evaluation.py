# Module Name: CVRPAdvantageEvaluation
# Description: Online evaluation for advantage functions during CVRP training.
#   Mode B: clones the current model, runs N training batches with candidate
#   and default advantage functions, and returns the delta reward improvement.
#
# This file is part of the LLM4AD + POMO integration project.

from __future__ import annotations

import sys
import os
import tempfile
from typing import Any

import numpy as np
import traceback
import torch

from llm4ad.base import Evaluation
from llm4ad.task.optimization.cvrp_advantage.template import template_program, task_description

__all__ = ['CVRPAdvantageEvaluation']

# ---------------------------------------------------------------------------
# Online evaluation parameters
# ---------------------------------------------------------------------------
N_EVAL_BATCHES = 10        # number of training batches for quick comparison
EVAL_BATCH_SIZE = 64       # batch size for evaluation
EVAL_SEED = 42             # fixed seed so candidate vs default see same data
# ---------------------------------------------------------------------------

# Resolve POMO root once at import time so the subprocess inherits it.
_POMO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '../../../../../POMO/NEW_py_ver')
)
_POMO_CVRP_DIR = os.path.join(_POMO_ROOT, 'CVRP')
_POMO_CVRP = os.path.join(_POMO_CVRP_DIR, 'POMO')
sys.path.insert(0, _POMO_ROOT)
sys.path.insert(0, _POMO_CVRP_DIR)
sys.path.insert(0, _POMO_CVRP)


class CVRPAdvantageEvaluation(Evaluation):
    """Online evaluator: compares candidate advantage vs default on current model.

    The trainer calls :meth:`set_context` before triggering an EoH round so that
    every candidate function is scored by training a few batches from the same
    starting checkpoint and comparing the trajectory reward against the standard
    POMO advantage.
    """

    def __init__(self,
                 timeout_seconds: int = 120,
                 **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            safe_evaluate=True,
            fork_proc=False,  # spawn: avoids CUDA+fork issues in subprocess
        )

        # Context set by the trainer before an EoH round.
        self._checkpoint_path: str | None = None
        self._problem_size: int = 100
        self._pomo_size: int = 100

    # ------------------------------------------------------------------
    #  Public API for the trainer
    # ------------------------------------------------------------------

    def set_context(self,
                    checkpoint_path: str,
                    problem_size: int,
                    pomo_size: int) -> None:
        """Store the current model checkpoint so evaluations are A/B-comparable.

        Call this *before* starting an EoH round so every candidate is compared
        from the same starting point.
        """
        self._checkpoint_path = checkpoint_path
        self._problem_size = problem_size
        self._pomo_size = pomo_size

    # ------------------------------------------------------------------
    #  LLM4AD interface
    # ------------------------------------------------------------------

    def evaluate_program(self,
                         program_str: str,
                         callable_func: callable,
                         **kwargs) -> Any | None:
        """Run N batches with candidate vs default advantage, return delta reward.

        Positive delta → candidate produces better trajectory rewards.
        """
        if self._checkpoint_path is None:
            return None

        from CVRPTrainer import CVRPTrainer as Trainer

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(self._checkpoint_path, map_location=device,
                                weights_only=False)

        # ── Candidate ──────────────────────────────────────────────
        torch.manual_seed(EVAL_SEED)
        score_cand = self._train_n_batches(
            checkpoint, callable_func)
        if score_cand is None:
            return None

        # ── Default ────────────────────────────────────────────────
        torch.manual_seed(EVAL_SEED)
        score_def = self._train_n_batches(
            checkpoint, Trainer._default_advantage)
        if score_def is None:
            return None

        # score = route distance, lower is better.
        # Return positive delta when candidate beats default.
        return float(score_def - score_cand)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _train_n_batches(self,
                         checkpoint: dict,
                         advantage_fn: callable) -> float | None:
        """Create a fresh trainer from *checkpoint*, run N batches, return
        the average training score of the last few batches."""
        from CVRPTrainer import CVRPTrainer as Trainer
        from utils.utils import set_result_folder

        tmpdir = tempfile.mkdtemp(prefix='cvrp_eval_')
        set_result_folder(tmpdir)

        try:
            trainer = self._build_eval_trainer()
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.advantage_fn = advantage_fn

            batch_scores = []
            for _ in range(N_EVAL_BATCHES):
                score, _ = trainer._train_one_batch(EVAL_BATCH_SIZE, epoch=1)
                batch_scores.append(score)

            # Use the last 3 batches for stability
            return float(np.mean(batch_scores[-3:]))
        except Exception:
            traceback.print_exc()
            return None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _build_eval_trainer(self):
        """Build a minimal CVRPTrainer configured for quick evaluation."""
        from CVRPTrainer import CVRPTrainer as Trainer

        env_params = {
            'problem_size': self._problem_size,
            'pomo_size': self._pomo_size,
        }
        model_params = {
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128 ** 0.5,
            'encoder_layer_num': 6,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
        }
        optimizer_params = {
            'optimizer': {'lr': 1e-4, 'weight_decay': 1e-6},
            'scheduler': {'milestones': [8001, 8051], 'gamma': 0.1},
        }
        _use_cuda = torch.cuda.is_available()
        trainer_params = {
            'use_cuda': _use_cuda,
            'cuda_device_num': 0 if _use_cuda else None,
            'epochs': 0,
            'train_episodes': N_EVAL_BATCHES * EVAL_BATCH_SIZE,
            'train_batch_size': EVAL_BATCH_SIZE,
            'logging': {
                'model_save_interval': 999999,
                'img_save_interval': 999999,
                'log_image_params_1': {
                    'json_foldername': 'log_image_style',
                    'filename': 'style_cvrp_100.json',
                },
                'log_image_params_2': {
                    'json_foldername': 'log_image_style',
                    'filename': 'style_loss_1.json',
                },
            },
            'model_load': {'enable': False},
        }

        return Trainer(
            env_params=env_params,
            model_params=model_params,
            optimizer_params=optimizer_params,
            trainer_params=trainer_params,
        )
