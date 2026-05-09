# Module Name: SearchController
# Description: Meta-controller for online EoH search during CVRP training.
#   Decides *when* to search, with what *intensity*, and which *operators* to
#   prioritise based on training metrics and a history of past search outcomes.
#
# This file is part of the LLM4AD + POMO integration project.

from __future__ import annotations

import dataclasses
import logging
import os
import re
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SearchRecord:
    """One row in the function library – records a past search and its outcome."""

    trigger_epoch: int
    pre_switch_score: float           # train score just before search
    search_intensity: str             # "light" | "medium" | "heavy"
    sample_count: int                 # how many candidates were tried
    operators: list[str]              # e.g. ["e1","e2","m1"]
    pop_size: int
    direction_hint: str               # free-text hint given to EoH
    best_delta: float                 # best candidate score improvement
    effective: bool                   # whether the new function was adopted

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchRecord":
        return cls(**d)


@dataclasses.dataclass
class SearchDecision:
    """Hyper-parameter bundle returned by the controller."""

    search_intensity: str   # "light" | "medium" | "heavy"
    sample_count: int       # number of EoH candidates
    operators: list[str]    # which EoH operators to enable
    pop_size: int           # EoH population size
    direction_hint: str     # search bias fed into task_description


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class SearchController:
    """LLM-driven meta-controller for online advantage-function search.

    Usage sketch::

        controller = SearchController(llm)

        # --- in the training loop ---
        if should_search(metrics):
            decision = controller.decide(
                epoch=epoch,
                recent_scores=[...],
                recent_losses=[...],
                plateau_epochs=15,
                total_epochs=8100,
            )
            # ... run EoH with *decision* ...
            controller.record(SearchRecord(...))
    """

    # Defaults used when LLM is unavailable or response is unparseable.
    _FALLBACK = SearchDecision(
        search_intensity="medium",
        sample_count=20,
        operators=["e1", "e2", "m1", "m2"],
        pop_size=4,
        direction_hint="",
    )

    def __init__(self, llm, log_dir: str = None):
        self._llm = llm
        self.history: list[SearchRecord] = []

        # --- logging ---
        self._log_dir = log_dir
        self._logger = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._logger = logging.getLogger(
                f'SearchController_{id(self)}')
            self._logger.setLevel(logging.DEBUG)
            # file handler – one log file per training run
            log_path = os.path.join(
                log_dir,
                f'controller_{datetime.now():%Y%m%d_%H%M%S}.log')
            fh = logging.FileHandler(log_path, mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '[%(asctime)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'))
            self._logger.addHandler(fh)
            # also print to console
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                '[Controller] %(message)s'))
            self._logger.addHandler(ch)
            self._logger.info('SearchController initialized, '
                              'log file: %s', log_path)

    # ------------------------------------------------------------------
    #  Core API
    # ------------------------------------------------------------------

    def decide(self,
               epoch: int,
               recent_scores: list[float],
               recent_losses: list[float],
               plateau_epochs: int,
               total_epochs: int) -> SearchDecision:
        """Ask the LLM for search hyper-parameters given current training state.

        Falls back to ``_FALLBACK`` if the LLM call fails or the response
        cannot be parsed.
        """
        try:
            prompt = self._build_meta_prompt(
                epoch, recent_scores, recent_losses,
                plateau_epochs, total_epochs)

            if self._logger:
                self._logger.info(
                    'DECIDE epoch=%d plateau=%d '
                    'score_recent=%.4f→%.4f '
                    'history_size=%d',
                    epoch, plateau_epochs,
                    recent_scores[0], recent_scores[-1],
                    len(self.history))
                self._logger.debug('--- DECIDE PROMPT ---\n%s\n'
                                   '--- END PROMPT ---', prompt)

            response = self._llm.draw_sample(prompt)

            if self._logger:
                self._logger.debug('--- LLM RESPONSE ---\n%s\n'
                                   '--- END RESPONSE ---', response)

            decision = self._parse_decision(response)

            if self._logger:
                self._logger.info(
                    'DECISION intensity=%s samples=%d '
                    'ops=%s pop=%d hint=%r',
                    decision.search_intensity,
                    decision.sample_count,
                    ','.join(decision.operators),
                    decision.pop_size,
                    decision.direction_hint)

            return decision
        except Exception as exc:
            if self._logger:
                self._logger.warning(
                    'DECIDE failed (fallback): %s', exc)
            return SearchController._FALLBACK

    def record(self, entry: SearchRecord) -> None:
        """Append a completed search to the history library."""
        self.history.append(entry)
        if self._logger:
            self._logger.info(
                'RECORD epoch=%d intensity=%s samples=%d '
                'delta=%+.4f effective=%s history=%d',
                entry.trigger_epoch,
                entry.search_intensity,
                entry.sample_count,
                entry.best_delta,
                entry.effective,
                len(self.history))

    def should_search(self,
                      epoch: int,
                      recent_scores: list[float],
                      recent_losses: list[float],
                      plateau_epochs: int,
                      total_epochs: int,
                      reflections: str = '') -> bool:
        """Ask LLM whether to trigger an EoH search now. Returns True/False."""
        try:
            prompt = self._build_check_prompt(
                epoch, recent_scores, recent_losses,
                plateau_epochs, total_epochs, reflections)

            if self._logger:
                self._logger.info(
                    'CHECK epoch=%d plateau=%d '
                    'score=%.4f→%.4f history=%d',
                    epoch, plateau_epochs,
                    recent_scores[0], recent_scores[-1],
                    len(self.history))
                self._logger.debug('--- CHECK PROMPT ---\n%s\n'
                                   '--- END PROMPT ---', prompt)

            response = self._llm.draw_sample(prompt)

            if self._logger:
                self._logger.info('CHECK response: %r', response)

            return self._parse_yes_no(response)
        except Exception as exc:
            if self._logger:
                self._logger.warning('CHECK failed (default NO): %s', exc)
            return False

    # ------------------------------------------------------------------
    #  Prompt construction
    # ------------------------------------------------------------------

    def _build_check_prompt(self,
                            epoch: int,
                            recent_scores: list[float],
                            recent_losses: list[float],
                            plateau_epochs: int,
                            total_epochs: int,
                            reflections: str = '') -> str:
        """Prompt asking LLM YES/NO: should we search now?"""
        lines = [
            f"Training epoch: {epoch}/{total_epochs}",
            f"Score trend (lower=better): {self._format_series(recent_scores)}",
            f"Loss trend: {self._format_series(recent_losses)}",
            f"Plateau (epochs since last best): {plateau_epochs}",
            f"EoH searches so far: {len(self.history)}",
        ]
        if self.history:
            effective = sum(1 for r in self.history if r.effective)
            lines.append(f"  Effective switches: {effective}")
            last = self.history[-1]
            lines.append(
                f"  Last search: epoch {last.trigger_epoch}, "
                f"delta={last.best_delta:+.4f}, "
                f"{'effective' if last.effective else 'discarded'}")

        if reflections:
            lines.append(f"\nPast EoH experience:\n{reflections}")

        lines += [
            "",
            "Should we run an EoH search NOW to design a better advantage function?",
            "YES — if training seems stuck, or past searches were helpful.",
            "NO  — if score is still improving well, or past searches were ineffective.",
            "Reply exactly YES or NO (single word).",
        ]
        return '\n'.join(lines)

    @staticmethod
    def _parse_yes_no(response: str) -> bool:
        return response.strip().upper().startswith('YES')

    # ------------------------------------------------------------------
    #  Prompt construction (for decide)
    # ------------------------------------------------------------------

    def _build_meta_prompt(self,
                           epoch: int,
                           recent_scores: list[float],
                           recent_losses: list[float],
                           plateau_epochs: int,
                           total_epochs: int) -> str:
        """Build a structured prompt for the LLM meta-controller."""

        # -- 1. Training state snapshot --
        state_lines = [
            f"Current epoch: {epoch} / {total_epochs} ({100*epoch//total_epochs}%)",
            f"Plateau detected: {plateau_epochs} epochs without improvement",
            "(Note: train score = avg route distance; LOWER score = better routes)",
            f"Recent train scores (last {len(recent_scores)} epochs):",
            f"  {self._format_series(recent_scores)}",
            f"Recent train losses (last {len(recent_losses)} epochs):",
            f"  {self._format_series(recent_losses)}",
        ]

        # -- 2. Past search history --
        if self.history:
            history_lines = [
                f"Past searches ({len(self.history)} total):"
            ]
            for i, rec in enumerate(self.history, 1):
                status = "effective" if rec.effective else "discarded"
                history_lines.append(
                    f"  #{i} epoch={rec.trigger_epoch} "
                    f"intensity={rec.search_intensity} "
                    f"samples={rec.sample_count} "
                    f"ops={','.join(rec.operators)} "
                    f"delta={rec.best_delta:+.4f} "
                    f"status={status}"
                )
        else:
            history_lines = ["No past searches yet."]

        # -- 3. Decision request --
        decision_lines = [
            "",
            "Based on the above, decide the search hyper-parameters for this round.",
            "Reply with key:value pairs (one per line):",
            "  intensity: light|medium|heavy",
            "  sample_count: 10|20|30",
            "  operators: e1,e2,m1,m2 (comma-separated subset)",
            "  pop_size: 2-8",
            "  hint: a short direction for EoH (e.g. 'penalise unfinished trajectories')",
            "",
            "Do not include any other text.",
        ]

        prompt = "\n".join(state_lines + [""] + history_lines + decision_lines)
        return prompt

    # ------------------------------------------------------------------
    #  Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_decision(response: str) -> SearchDecision:
        """Extract key:value pairs from LLM response."""
        pairs: dict[str, str] = {}
        for line in response.strip().splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                pairs[key.strip().lower()] = value.strip()

        # --- intensity ---
        intensity = pairs.get("intensity", "medium")
        if intensity not in ("light", "medium", "heavy"):
            intensity = "medium"
        intensity_map = {"light": 10, "medium": 20, "heavy": 30}

        # --- sample_count (override by intensity if missing) ---
        try:
            sample_count = int(pairs.get("sample_count", intensity_map[intensity]))
        except (ValueError, KeyError):
            sample_count = intensity_map[intensity]
        sample_count = max(5, min(50, sample_count))

        # --- operators ---
        ops_str = pairs.get("operators", "e1,e2,m1,m2")
        operators = [o.strip() for o in ops_str.split(",")
                     if o.strip() in ("e1", "e2", "m1", "m2")] or ["e1", "e2"]

        # --- pop_size ---
        try:
            pop_size = int(pairs.get("pop_size", 4))
        except ValueError:
            pop_size = 4
        pop_size = max(2, min(10, pop_size))

        # --- hint ---
        hint = pairs.get("hint", "")

        return SearchDecision(
            search_intensity=intensity,
            sample_count=sample_count,
            operators=operators,
            pop_size=pop_size,
            direction_hint=hint,
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_series(values: list[float], max_items: int = 8) -> str:
        """Compact display of a numeric series (for the prompt)."""
        if len(values) <= max_items:
            return ", ".join(f"{v:.4f}" for v in values)

        # Show first 4 + last 4
        head = ", ".join(f"{v:.4f}" for v in values[:4])
        tail = ", ".join(f"{v:.4f}" for v in values[-4:])
        return f"{head}, ..., {tail}"
