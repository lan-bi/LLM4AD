#!/usr/bin/env python3
"""run_online_eoh.py — Online EoH-integrated CVRP POMO training.

This script replaces the standard CVRPTrainer.run() with a loop that
periodically triggers EoH searches to evolve better advantage functions
*while training continues*, using the current model state as the
evaluation baseline.

Usage:
    python run_online_eoh.py

Before running, edit the LLM credentials and problem configuration below.
"""

from __future__ import annotations

import os
import sys
import tempfile

# ---------------------------------------------------------------------------
#  Path setup — make POMO and LLM4AD importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POMO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR,
                                          '../../../../../POMO/NEW_py_ver'))
_POMO_CVRP_DIR = os.path.join(_POMO_ROOT, 'CVRP')
_POMO_CVRP = os.path.join(_POMO_CVRP_DIR, 'POMO')
sys.path.insert(0, _POMO_ROOT)
sys.path.insert(0, _POMO_CVRP_DIR)
sys.path.insert(0, _POMO_CVRP)
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, '../../../../')))

# ---------------------------------------------------------------------------
#  Configuration — edit these before running
# ---------------------------------------------------------------------------

# LLM credentials (same format as LLM4AD examples)
LLM_CONFIG = {
    'host': os.environ.get('LLM4AD_HOST', 'api.deepseek.com'),
    'key': os.environ['LLM4AD_KEY'],
    'model': os.environ.get('LLM4AD_MODEL', 'deepseek-chat'),
    'timeout': 60,
}

# Training configuration (mirrors train_n100.py defaults)
TRAIN_CONFIG = {
    'problem_size': 20,          # proxy: CVRP20; set to 100 for production
    'pomo_size': 20,             # = problem_size for POMO
    'epochs': 200,               # total training epochs
    'train_episodes': 2000,      # episodes per epoch
    'train_batch_size': 64,
    'use_cuda': True,
    'cuda_device_num': 0,
}

# Online EoH configuration
ONLINE_CONFIG = {
    'plateau_min_epochs': 30,    # trigger search after this many non-improving epochs
    'force_search_every': 100,   # force a search at least every N epochs
    'log_dir': './logs/online_eoh',
    'eval_timeout_seconds': 120,
}

# ---------------------------------------------------------------------------
#  Imports (after path setup)
# ---------------------------------------------------------------------------
import torch

from CVRPTrainer import CVRPTrainer as Trainer
from utils.utils import set_result_folder

from llm4ad.task.optimization.cvrp_advantage import (
    CVRPAdvantageEvaluation,
    SearchController,
    SearchRecord,
    SearchDecision,
    template_program,
    task_description,
)
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.base import TextFunctionProgramConverter


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _compile_function(best_function, template: str):
    """Compile a Function object into a callable using the template imports."""
    from llm4ad.base import Function as FuncType
    if best_function is None:
        return None

    # Convert Function → Program (adds imports from template)
    program = TextFunctionProgramConverter.function_to_program(
        best_function, template)
    if program is None:
        return None

    try:
        callables = program.exec()
        return callables[0]
    except Exception:
        return None


def _build_trainer():
    """Build a CVRPTrainer with the training configuration above."""
    env_params = {
        'problem_size': TRAIN_CONFIG['problem_size'],
        'pomo_size': TRAIN_CONFIG['pomo_size'],
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
        'use_cuda': TRAIN_CONFIG['use_cuda'],
        'cuda_device_num': TRAIN_CONFIG['cuda_device_num'],
        'epochs': TRAIN_CONFIG['epochs'],
        'train_episodes': TRAIN_CONFIG['train_episodes'],
        'train_batch_size': TRAIN_CONFIG['train_batch_size'],
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


def _run_eoh_and_switch(llm, evaluation, controller,
                        trainer, epoch: int, decision: SearchDecision):
    """Run one EoH round, compile the best function, and switch if effective."""
    log_dir = os.path.join(ONLINE_CONFIG['log_dir'],
                           f'eoh_epoch{epoch}')
    profiler = EoHProfiler(log_dir=log_dir, log_style='simple')

    eoh = EoH(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        max_sample_nums=decision.sample_count,
        pop_size=decision.pop_size,
        use_e2_operator='e2' in decision.operators,
        use_m1_operator='m1' in decision.operators,
        use_m2_operator='m2' in decision.operators,
        num_samplers=1,
        num_evaluators=1,
        debug_mode=False,
    )
    eoh.run()

    best_fn = profiler._cur_best_function
    best_score = profiler._cur_best_program_score

    if best_fn is None or best_score is None or best_score <= 0:
        # No improvement over default advantage
        controller.record(SearchRecord(
            trigger_epoch=epoch,
            pre_switch_score=trainer.score_history[-1]
            if trainer.score_history else 0.0,
            search_intensity=decision.search_intensity,
            sample_count=decision.sample_count,
            operators=decision.operators,
            pop_size=decision.pop_size,
            direction_hint=decision.direction_hint,
            best_delta=float(best_score) if best_score else 0.0,
            effective=False,
        ))
        return

    callable_fn = _compile_function(best_fn, template_program)
    if callable_fn is None:
        return

    trainer.switch_advantage(callable_fn)

    controller.record(SearchRecord(
        trigger_epoch=epoch,
        pre_switch_score=trainer.score_history[-1]
        if trainer.score_history else 0.0,
        search_intensity=decision.search_intensity,
        sample_count=decision.sample_count,
        operators=decision.operators,
        pop_size=decision.pop_size,
        direction_hint=decision.direction_hint,
        best_delta=float(best_score),
        effective=True,
    ))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ONLINE_CONFIG['log_dir'], exist_ok=True)
    set_result_folder(ONLINE_CONFIG['log_dir'])

    # --- build components ---
    llm = HttpsApi(**LLM_CONFIG)
    trainer = _build_trainer()
    controller = SearchController(
        llm, log_dir=os.path.join(ONLINE_CONFIG['log_dir'], 'controller'))
    evaluation = CVRPAdvantageEvaluation(
        timeout_seconds=ONLINE_CONFIG['eval_timeout_seconds'])

    trainer.logger.info('=== Online EoH-integrated training started ===')
    trainer.logger.info('Problem: CVRP%d, Epochs: %d',
                        TRAIN_CONFIG['problem_size'],
                        TRAIN_CONFIG['epochs'])

    last_search_epoch = 0
    trainer.time_estimator.reset(trainer.start_epoch)

    for epoch in range(trainer.start_epoch, TRAIN_CONFIG['epochs'] + 1):
        trainer.logger.info('=' * 65)

        # LR decay
        trainer.scheduler.step()

        # Train one epoch
        train_score, train_loss = trainer._train_one_epoch(epoch)
        trainer.result_log.append('train_score', epoch, train_score)
        trainer.result_log.append('train_loss', epoch, train_loss)

        # Update history + plateau tracking
        trainer.score_history.append(train_score)
        trainer.loss_history.append(train_loss)
        if train_score > trainer.best_score:
            trainer.best_score = train_score
            trainer.plateau_counter = 0
        else:
            trainer.plateau_counter += 1

        # --- Search trigger check ---
        epochs_since_search = epoch - last_search_epoch
        plateau_trigger = trainer.detect_plateau(
            ONLINE_CONFIG['plateau_min_epochs'])
        periodic_trigger = (
            epochs_since_search >= ONLINE_CONFIG['force_search_every']
            and last_search_epoch > 0)

        if plateau_trigger or periodic_trigger:
            trigger_reason = 'plateau' if plateau_trigger else 'periodic'
            trainer.logger.info(
                'Triggering EoH search (reason=%s, epoch=%d, plateau=%d epochs)',
                trigger_reason, epoch, trainer.plateau_counter)

            # Save checkpoint for evaluation
            ckpt_path = os.path.join(
                ONLINE_CONFIG['log_dir'],
                f'temp_ckpt_epoch{epoch}.pt')
            trainer.save_temp_checkpoint(ckpt_path)
            evaluation.set_context(
                ckpt_path,
                TRAIN_CONFIG['problem_size'],
                TRAIN_CONFIG['pomo_size'])

            # Controller decides hyperparams
            decision = controller.decide(
                epoch=epoch,
                recent_scores=trainer.score_history[-50:],
                recent_losses=trainer.loss_history[-50:],
                plateau_epochs=trainer.plateau_counter,
                total_epochs=TRAIN_CONFIG['epochs'],
            )

            # Run EoH and switch if effective
            _run_eoh_and_switch(
                llm, evaluation, controller, trainer, epoch, decision)

            last_search_epoch = epoch
            os.remove(ckpt_path)

        # Logging
        elapsed, remain = trainer.time_estimator.get_est_string(
            epoch, TRAIN_CONFIG['epochs'])
        trainer.logger.info(
            'Epoch %3d/%-3d  Score: %.4f  Loss: %.4f  '
            'Plateau: %d  Best: %.4f  Elapsed[%s] Remain[%s]',
            epoch, TRAIN_CONFIG['epochs'],
            train_score, train_loss,
            trainer.plateau_counter, trainer.best_score,
            elapsed, remain)

    trainer.logger.info('=== Training complete ===')
    trainer.logger.info('Total EoH searches: %d', len(controller.history))
    trainer.logger.info('Effective switches: %d',
                        sum(1 for r in controller.history if r.effective))


if __name__ == '__main__':
    main()
