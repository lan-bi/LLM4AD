# Module Name: CVRPAdvantageEvaluation
# Description: Evaluation class for evolving advantage functions
#   for CVRP POMO training.  Runs lightweight proxy training (CVRP20)
#   and returns final training score as fitness.
#
# This file is part of the LLM4AD + POMO integration project.

from __future__ import annotations

import sys
import os
import tempfile
from typing import Any

import torch

from llm4ad.base import Evaluation
from llm4ad.task.optimization.cvrp_advantage.template import template_program, task_description

__all__ = ['CVRPAdvantageEvaluation']

# ---------------------------------------------------------------------------
# Proxy training configuration
# ---------------------------------------------------------------------------
PROXY_PROBLEM_SIZE = 20
PROXY_POMO_SIZE = 20
PROXY_EPOCHS = 5
PROXY_TRAIN_EPISODES = 1000
PROXY_TRAIN_BATCH_SIZE = 64
PROXY_SEEDS = [1234, 5678]
# ---------------------------------------------------------------------------

# Resolve POMO root once at import time so the subprocess inherits it.
_POMO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '../../../../../../POMO/NEW_py_ver')
)
_POMO_CVRP = os.path.join(_POMO_ROOT, 'CVRP', 'POMO')
sys.path.insert(0, _POMO_ROOT)
sys.path.insert(0, _POMO_CVRP)


class CVRPAdvantageEvaluation(Evaluation):
    """Runs proxy CVRP POMO training to score an advantage function."""

    def __init__(self, timeout_seconds: int = 300, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            safe_evaluate=True,
            fork_proc='auto',
        )

    # ------------------------------------------------------------------
    #  LLM4AD interface
    # ------------------------------------------------------------------

    def evaluate_program(self,
                         program_str: str,
                         callable_func: callable,
                         **kwargs) -> Any | None:
        """
        Train CVRP20 for PROXY_EPOCHS with *callable_func* as the
        advantage function, averaged over PROXY_SEEDS.

        Returns:
            float – negative of final mean training score (higher is better).
        """
        scores = []
        for seed in PROXY_SEEDS:
            score = self._run_one_seed(callable_func, seed)
            if score is None:
                return None
            scores.append(score)

        # Fitness = -(final training score)  →  minimise distance
        return -float(sum(scores) / len(scores))

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_one_seed(advantage_fn, seed) -> float | None:
        """Run one complete proxy training and return final train_score."""
        from CVRPTrainer import CVRPTrainer as Trainer
        from utils.utils import set_result_folder

        torch.manual_seed(seed)

        # Use a temp directory so each evaluation does not pollute disk.
        tmpdir = tempfile.mkdtemp(prefix='cvrp_ahd_eval_')
        set_result_folder(tmpdir)

        try:
            trainer = CVRPAdvantageEvaluation._build_trainer()
            trainer.advantage_fn = advantage_fn
            trainer.run()

            # Extract final epoch training score from LogData
            return trainer.result_log.get('train_score')[-1]
        finally:
            # Clean up temp directory (checkpoints / images were
            # suppressed by high intervals, but remove any that snuck in).
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _build_trainer():
        from CVRPTrainer import CVRPTrainer as Trainer

        env_params = {
            'problem_size': PROXY_PROBLEM_SIZE,
            'pomo_size': PROXY_POMO_SIZE,
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
        trainer_params = {
            'use_cuda': True,
            'cuda_device_num': 0,
            'epochs': PROXY_EPOCHS,
            'train_episodes': PROXY_TRAIN_EPISODES,
            'train_batch_size': PROXY_TRAIN_BATCH_SIZE,
            'logging': {
                'model_save_interval': 9999,   # never save during proxy
                'img_save_interval': 9999,
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
