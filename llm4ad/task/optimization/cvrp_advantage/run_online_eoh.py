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

import json
import os
import re
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
    'problem_size': 100,         # CVRP100
    'pomo_size': 100,            # = problem_size for POMO
    'epochs': 1000,              # total training epochs
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
    'design_review_interval': 3,  # run design review every N EoH searches
}

# ---------------------------------------------------------------------------
#  Imports (after path setup)
# ---------------------------------------------------------------------------
import torch
from torch.utils.tensorboard import SummaryWriter

from CVRPTrainer import CVRPTrainer as Trainer
from utils.utils import create_logger, set_result_folder

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

class _KeyMaskedLLM:
    """Wrap an LLM to mask ``_key`` from profiler ``__dict__`` inspection."""

    def __init__(self, llm):
        self._wrapped = llm

    def __getattr__(self, name):
        if name == '__dict__':
            d = self._wrapped.__dict__.copy()
            if '_key' in d:
                d['_key'] = '***'
            return d
        return getattr(self._wrapped, name)


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

    # Wrap LLM so profiler log never sees the raw API key
    eoh = EoH(
        llm=_KeyMaskedLLM(llm),
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
        return False, None

    callable_fn = _compile_function(best_fn, template_program)
    if callable_fn is None:
        return False, None

    trainer.switch_advantage(callable_fn)
    fn_source = str(best_fn)  # serialise for resume

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
    return True, fn_source


# ---------------------------------------------------------------------------
#  Reflection agent – learns from evaluation failures
# ---------------------------------------------------------------------------

_REFLECTION_FILE = 'reflections.json'
_MAX_ERRORS_PER_REFLECTION = 12


def _collect_eoh_errors(eoh_log_dir: str) -> dict:
    """Scan EoH profiler logs for failure stats + traceback excerpts."""
    info = {'total': 0, 'valid': 0, 'errors': []}
    for entry in sorted(os.listdir(eoh_log_dir)):
        inner = os.path.join(eoh_log_dir, entry)
        if not os.path.isdir(inner):
            continue
        samples_dir = os.path.join(inner, 'samples')
        if not os.path.isdir(samples_dir):
            continue
        for fname in sorted(os.listdir(samples_dir)):
            if not (fname.startswith('samples_') and fname.endswith('.json')):
                continue
            with open(os.path.join(samples_dir, fname)) as f:
                data = json.load(f)
            for sample in data:
                info['total'] += 1
                if sample.get('score') is not None:
                    info['valid'] += 1
    for entry in sorted(os.listdir(eoh_log_dir)):
        inner = os.path.join(eoh_log_dir, entry)
        if not os.path.isdir(inner):
            continue
        log_path = os.path.join(inner, 'run_log.txt')
        if not os.path.exists(log_path):
            continue
        with open(log_path, errors='replace') as f:
            text = f.read()
        tb_blocks = re.findall(
            r'(Traceback \(most recent call last\):.*?)(?=\n\n|\nSample|\Z)',
            text, re.DOTALL)
        for block in tb_blocks:
            lines = block.strip().splitlines()
            short = '\n'.join(lines[-4:]) if len(lines) > 4 else block
            info['errors'].append(short)
    info['errors'] = info['errors'][:_MAX_ERRORS_PER_REFLECTION]
    info['failed'] = info['total'] - info['valid']
    return info


def _call_reflection_agent(llm, error_info: dict, prev_lessons: str) -> str:
    """Ask LLM to analyse errors → return bullet-point design lessons."""
    if error_info['total'] == 0 or error_info['failed'] == 0:
        return prev_lessons or ''
    parts = [
        "Analyse runtime errors from a CVRP advantage-function search.",
        f"Latest round: {error_info['total']} samples, "
        f"{error_info['valid']} valid, {error_info['failed']} failed.",
    ]
    if error_info['errors']:
        parts.append("Sample tracebacks (last lines):")
        for i, err in enumerate(error_info['errors'][:8], 1):
            parts.append(f"  #{i}: {err}")
    if prev_lessons:
        parts.append(f"\nPrevious lessons:\n{prev_lessons}")
    parts += [
        "",
        "Identify NEW mistake patterns. Output 2-5 concise bullets (each '- ').",
        "If no new patterns, output exactly 'NO_NEW_LESSONS'.",
    ]
    prompt = '\n'.join(parts)
    try:
        response = llm.draw_sample(prompt)
    except Exception:
        response = 'NO_NEW_LESSONS'
    if 'NO_NEW_LESSONS' in response:
        return prev_lessons or ''
    if prev_lessons:
        return prev_lessons.strip() + '\n' + response.strip()
    return response.strip()


def _load_reflections(state_dir: str) -> str:
    path = os.path.join(state_dir, _REFLECTION_FILE)
    if not os.path.exists(path):
        return ''
    with open(path) as f:
        return json.load(f).get('lessons', '')


def _save_reflections(state_dir: str, lessons: str) -> None:
    with open(os.path.join(state_dir, _REFLECTION_FILE), 'w') as f:
        json.dump({'lessons': lessons}, f, indent=2)


def _augment_task_description(base: str, reflections: str) -> str:
    """Append accumulated design lessons to the task description."""
    if not reflections:
        return base
    return (base
            + '\n\n## Lessons Learned from Previous Attempts\n'
            + 'The following patterns caused runtime errors.  Avoid them:\n\n'
            + reflections)


# ---------------------------------------------------------------------------
#  Design review – periodic analysis of good vs bad functions
# ---------------------------------------------------------------------------

def _collect_best_worst_functions(log_root: str, top_k: int = 2
                                   ) -> tuple[list, list]:
    """Scan all EoH round logs, return (best_funcs, worst_funcs)."""
    all_funcs = []  # (score, func_str)
    for entry in sorted(os.listdir(log_root)):
        if not entry.startswith('eoh_epoch'):
            continue
        eoh_dir = os.path.join(log_root, entry)
        for inner in sorted(os.listdir(eoh_dir)):
            samples_dir = os.path.join(eoh_dir, inner, 'samples')
            if not os.path.isdir(samples_dir):
                continue
            for fname in sorted(os.listdir(samples_dir)):
                if not (fname.startswith('samples_') and fname.endswith('.json')):
                    continue
                with open(os.path.join(samples_dir, fname)) as f:
                    data = json.load(f)
                for sample in data:
                    s = sample.get('score')
                    if s is None:
                        continue
                    fn = sample.get('function', '')
                    all_funcs.append((s, fn))
    if not all_funcs:
        return [], []
    all_funcs.sort(key=lambda x: x[0])
    worst = all_funcs[:top_k]
    best = all_funcs[-top_k:]
    return best[::-1], worst


_TRUNC_FN_LENGTH = 600


def _call_design_review_agent(llm, best_funcs: list, worst_funcs: list,
                              prev_design_lessons: str) -> str:
    """Ask LLM to compare best vs worst functions → design patterns."""
    if not best_funcs and not worst_funcs:
        return prev_design_lessons or ''
    parts = ["Compare the best and worst CVRP advantage functions below.",
             "Identify what patterns make the good functions better."]
    if best_funcs:
        parts.append("\nTop-scoring functions:")
        for i, (score, fn) in enumerate(best_funcs, 1):
            parts.append(f"  #{i} (score={score:+.4f}):\n    {fn[:_TRUNC_FN_LENGTH]}")
    if worst_funcs:
        parts.append("\nLowest-scoring valid functions:")
        for i, (score, fn) in enumerate(worst_funcs, 1):
            parts.append(f"  #{i} (score={score:+.4f}):\n    {fn[:_TRUNC_FN_LENGTH]}")
    if prev_design_lessons:
        parts.append(f"\nPrevious design lessons:\n{prev_design_lessons}")
    parts += [
        "",
        "Output 2-4 concise bullet points (each '- ') of actionable design advice.",
        "If no new patterns, output exactly 'NO_NEW_LESSONS'.",
    ]
    prompt = '\n'.join(parts)
    try:
        response = llm.draw_sample(prompt)
    except Exception:
        response = 'NO_NEW_LESSONS'
    if 'NO_NEW_LESSONS' in response:
        return prev_design_lessons or ''
    if prev_design_lessons:
        return prev_design_lessons.strip() + '\n' + response.strip()
    return response.strip()


def _load_design_lessons(state_dir: str) -> str:
    path = os.path.join(state_dir, _REFLECTION_FILE)
    if not os.path.exists(path):
        return ''
    with open(path) as f:
        return json.load(f).get('design_lessons', '')


def _save_design_lessons(state_dir: str, design_lessons: str) -> None:
    path = os.path.join(state_dir, _REFLECTION_FILE)
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data['design_lessons'] = design_lessons
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _augment_task_description_full(base: str, error_lessons: str,
                                   design_lessons: str) -> str:
    """Append both error-prevention and design-pattern lessons."""
    result = base
    if error_lessons:
        result += ('\n\n## Error Prevention (lessons from runtime failures)\n'
                   + 'Avoid these patterns:\n\n' + error_lessons)
    if design_lessons:
        result += ('\n\n## Design Patterns That Work (lessons from scoring)\n'
                   + 'Good functions tend to:\n\n' + design_lessons)
    return result


# ---------------------------------------------------------------------------
#  Full-state persistence (for resume)
# ---------------------------------------------------------------------------

_FULL_STATE_FILE = 'full_state.json'


def _save_full_state(state_dir: str,
                     controller: SearchController,
                     advantage_fn_source: str | None,
                     last_search_epoch: int,
                     start_epoch: int) -> None:
    """Persist all EoH state to a JSON file for later resume."""
    data = {
        'controller_history': [r.to_dict() for r in controller.history],
        'advantage_fn_source': advantage_fn_source,
        'last_search_epoch': last_search_epoch,
        'start_epoch': start_epoch,
    }
    path = os.path.join(state_dir, _FULL_STATE_FILE)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _load_full_state(state_dir: str) -> dict | None:
    """Load persisted EoH state.  Returns None if no state file exists."""
    path = os.path.join(state_dir, _FULL_STATE_FILE)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def _compile_from_source(source: str | None, template: str):
    """Compile a serialised advantage function back into a callable.

    ``source`` should be the full function definition (including ``def``
    line).  If ``None``, return ``None`` (meaning use default).
    """
    if source is None:
        return None
    from llm4ad.base import TextFunctionProgramConverter
    func = TextFunctionProgramConverter.text_to_function(source)
    if func is None:
        return None
    program = TextFunctionProgramConverter.function_to_program(func, template)
    if program is None:
        return None
    try:
        callables = program.exec()
        return callables[0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ONLINE_CONFIG['log_dir'], exist_ok=True)
    set_result_folder(ONLINE_CONFIG['log_dir'])
    create_logger(log_file={'desc': 'online_eoh', 'filename': 'run_log'})

    # --- build components ---
    llm = HttpsApi(**LLM_CONFIG)
    trainer = _build_trainer()
    controller = SearchController(
        llm, log_dir=os.path.join(ONLINE_CONFIG['log_dir'], 'controller'))
    evaluation = CVRPAdvantageEvaluation(
        timeout_seconds=ONLINE_CONFIG['eval_timeout_seconds'])

    # --- TensorBoard ---
    tb_dir = os.path.join(ONLINE_CONFIG['log_dir'], 'tensorboard')
    writer = SummaryWriter(log_dir=tb_dir)

    # --- resume from checkpoint if available ---
    full_state = _load_full_state(ONLINE_CONFIG['log_dir'])
    last_search_epoch = 0
    current_adv_source: str | None = None  # None = use default

    if full_state is not None:
        # Restore EoH controller history
        for rec_dict in full_state.get('controller_history', []):
            controller.history.append(SearchRecord.from_dict(rec_dict))

        # Restore advantage function
        current_adv_source = full_state.get('advantage_fn_source')
        if current_adv_source is not None:
            compiled = _compile_from_source(
                current_adv_source, template_program)
            if compiled is not None:
                trainer.switch_advantage(compiled)

        last_search_epoch = full_state.get('last_search_epoch', 0)
        start_epoch = full_state.get('start_epoch', 1)
        trainer.start_epoch = start_epoch

        # Restore EoH state from the last model checkpoint
        resume_ckpt = os.path.join(ONLINE_CONFIG['log_dir'],
                                   'resume_checkpoint.pt')
        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location='cpu',
                              weights_only=False)
            trainer.model.load_state_dict(ckpt['model_state_dict'])
            trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            trainer.restore_eoh_state(ckpt)

        trainer.logger.info(
            '=== Resumed from checkpoint (epoch %d, %d controller records) ===',
            start_epoch, len(controller.history))
        # Apply any persisted reflections to the evaluation
        err_lessons = _load_reflections(ONLINE_CONFIG['log_dir'])
        design_lessons = _load_design_lessons(ONLINE_CONFIG['log_dir'])
        if err_lessons or design_lessons:
            evaluation._task_description = _augment_task_description_full(
                task_description, err_lessons, design_lessons)
    else:
        trainer.logger.info('=== Online EoH-integrated training started ===')

    trainer.logger.info('Problem: CVRP%d, Epochs: %d',
                        TRAIN_CONFIG['problem_size'],
                        TRAIN_CONFIG['epochs'])

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
            switched, fn_source = _run_eoh_and_switch(
                llm, evaluation, controller, trainer, epoch, decision)

            if switched:
                current_adv_source = fn_source

            # --- TensorBoard EoH events ---
            writer.add_scalar('EoH/TriggerEpoch', epoch, len(controller.history))
            writer.add_scalar('EoH/BestDelta',
                controller.history[-1].best_delta if controller.history else 0,
                len(controller.history))
            writer.add_scalar('EoH/Effective',
                1 if (controller.history and controller.history[-1].effective) else 0,
                len(controller.history))
            intensity_val = {'light': 1, 'medium': 2, 'heavy': 3}.get(
                decision.search_intensity, 2)
            writer.add_scalar('EoH/Intensity', intensity_val, len(controller.history))

            # --- reflection: learn from evaluation failures ---
            eoh_log = os.path.join(ONLINE_CONFIG['log_dir'],
                                   f'eoh_epoch{epoch}')
            err_info = _collect_eoh_errors(eoh_log)
            old_err = _load_reflections(ONLINE_CONFIG['log_dir'])
            new_err = _call_reflection_agent(llm, err_info, old_err)
            if new_err != old_err:
                _save_reflections(ONLINE_CONFIG['log_dir'], new_err)

            # --- design review: periodic analysis of best/worst ---
            n_searches = len(controller.history)
            interval = ONLINE_CONFIG['design_review_interval']
            old_design = _load_design_lessons(ONLINE_CONFIG['log_dir'])
            if n_searches > 0 and n_searches % interval == 0:
                best_fns, worst_fns = _collect_best_worst_functions(
                    ONLINE_CONFIG['log_dir'])
                new_design = _call_design_review_agent(
                    llm, best_fns, worst_fns, old_design)
                if new_design != old_design:
                    _save_design_lessons(ONLINE_CONFIG['log_dir'], new_design)
                    old_design = new_design

            evaluation._task_description = _augment_task_description_full(
                task_description, new_err, old_design)

            last_search_epoch = epoch

            # Persist full state for resume
            trainer.save_temp_checkpoint(
                os.path.join(ONLINE_CONFIG['log_dir'],
                             'resume_checkpoint.pt'))
            _save_full_state(
                ONLINE_CONFIG['log_dir'],
                controller, current_adv_source,
                last_search_epoch, epoch + 1)

            os.remove(ckpt_path)

        # TensorBoard training curves
        writer.add_scalar('Train/Score', train_score, epoch)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Plateau', trainer.plateau_counter, epoch)
        writer.add_scalar('Train/BestScore', trainer.best_score, epoch)

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

    writer.close()
    trainer.logger.info('=== Training complete ===')
    trainer.logger.info('Total EoH searches: %d', len(controller.history))
    trainer.logger.info('Effective switches: %d',
                        sum(1 for r in controller.history if r.effective))


if __name__ == '__main__':
    main()
