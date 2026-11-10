"""Generate BO algorithms with installed LLaMEA and resumable small (mu+lambda) search.

CLI:
  python -m runs.stage1 --target bbob --limit 3
  python -m runs.stage1 --target bbob --resume --limit 100

`--limit` is the TOTAL candidate count, not additional candidates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import warnings
from pathlib import Path
import numpy as np

# Pickle should reference a stable module name, not __main__, even when the
# entry point is invoked using `calibration_objective -m runs.stage1`.
if __name__ == '__main__':
    sys.modules['runs.stage1'] = sys.modules[__name__]

from src.problems import from_config, metric_for_target
from src.performance import Algorithm, evaluate_many
from src.utils import (DEFAULT_CONFIG, RESULTS, dump_json, group_seeds,
                       load_config, metadata, plot_fitness, seed_search)

# Installed library: it handles prompts, LLM calls, ES selection and lineage.
# Only the checkpoint writer is overridden below (atomic file replacement).
try:
    from llamea import LLaMEA
except ImportError:
    LLaMEA = None


class AtomicLLaMEA(LLaMEA if LLaMEA is not None else object):
    """LLaMEA with an atomic ES snapshot after every completed generation.

    Input: normal LLaMEA state. Output: trusted `llamea_config.pkl` archive.
    Saving a partial pickle must never destroy the previous committed state.
    """
    def initialize(self):
        """Checkpoint the entire initial parent population, including short runs."""
        super().initialize()
        self.pickle_archive()

    def _select_parents(self, count=None):
        """Do not generate a full offspring batch past the candidate limit.

        LLaMEA 1.4.2 checks the budget before each generation, but normally
        generates all n_offspring even if only one candidate slot remains.
        Keep its native selection/evolution code; shrink only the final batch.
        An explicit count is used for multi-parent operators, not batching.
        """
        if count is None:
            remaining = self.budget - len(self.run_history)
            count = min(self.n_offspring, remaining)
        return super()._select_parents(count=count)

    def pickle_archive(self):
        if not self.logger:
            raise RuntimeError('LLaMEA logging must be enabled for resume')
        folder = Path(self.logger.dirname)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'llamea_config.pkl'
        temporary = folder / 'llamea_config.pkl.tmp'
        # Restore these states on warm start for reproducible ES-side choices.
        self._checkpoint_random_state = random.getstate()
        self._checkpoint_numpy_state = np.random.get_state()
        try:
            with temporary.open('wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        # The candidate becomes COMMITTED only after LLaMEA has archived
        # the population, history and associated evaluation callback.
        self.f.committed(len(self.run_history), folder.resolve())


def make_llm(cfg):
    """Input: llm settings. Output: installed LLaMEA provider adapter."""
    from llamea import Gemini_LLM, Ollama_LLM
    options = cfg['llm']
    if options['provider'] == 'gemini':
        key = os.environ.get('GEMINI_API_KEY')
        if not key:
            raise RuntimeError('Set GEMINI_API_KEY in your shell (never in YAML)')
        return Gemini_LLM(model=options['model'], api_key=key)
    if options['provider'] == 'ollama':
        return Ollama_LLM(model=options['model'])
    raise ValueError('llm.provider must be gemini or ollama')


class CandidateScorer:
    """LLaMEA callback: Solution(code/name) -> Solution(mean AOCC, feedback).

    The LLM's completed code is first stored in code/candidate_NNN.py.
    A worker compiles it, imports the named class, and runs it using only a
    guarded `func(x)` callback to a fresh black-box problem per seed.
    Candidate phases are saved atomically; only committed ones enter ES history.
    """
    def __init__(self, cfg, target, folder, metric):
        self.cfg = cfg
        self.target = target
        self.folder = Path(folder).resolve()
        self.problem = from_config(cfg, target)
        self.metric = metric
        self.records = []

    def __call__(self, solution, explogger=None):
        n = len(self.records) + 1
        maximum = int(self.cfg['stage1']['max_candidates'])
        if n > maximum:
            raise RuntimeError('100-candidate limit reached')
        code = getattr(solution, 'code', '') or ''
        name = getattr(solution, 'name', '') or ''
        source = self.folder / 'code' / f'candidate_{n:03d}.py'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(code, encoding='utf-8')
        self._state(n, 'code_generated', source=str(source.relative_to(self.folder)))

        runs = []
        evaluation_error = ''
        if code and name:
            self._state(n, 'evaluating')
            try:
                stage = self.cfg['stage1']
                runs = evaluate_many(
                    Algorithm(name, code=code, class_name=name),
                    self.problem, group_seeds(self.cfg), int(stage['budget']),
                    timeout=float(stage['timeout_seconds']),
                    workers=int(stage['seed_workers']), aocc_config=self.metric,
                )
            except Exception as exc:
                evaluation_error = f'{type(exc).__name__}: {exc}'
                print(f'Candidate {n} evaluation failed: {evaluation_error}', file=sys.stderr)
        good = len(runs) == 10 and all(r.status == 'ok' and r.aocc is not None for r in runs)
        score = float(np.mean([r.aocc for r in runs])) if good else -1.0
        feedback = f'Fitness (mean AOCC across 10 runs): {score:.6f}; valid={good}'
        if good:
            final = float(np.mean([r.best for r in runs]))
            initial = float(np.mean([min(r.ys[:min(8, len(r.ys))]) for r in runs]))
            feedback += f'; mean initial best={initial:.6g}; mean final best={final:.6g}'
        if not good:
            # The previous code sent the FIRST 180 traceback characters (mostly
            # worker paths), hiding the actual exception at the END. Supply the
            # generated-code location and final exception to guide LLM repair.
            failed = next((r for r in runs if r.status != 'ok'), None)
            if failed:
                lines = [line.strip() for line in failed.error.splitlines() if line.strip()]
                location = next((line for line in reversed(lines)
                                 if '<generated candidate>' in line), '')
                reason = lines[-1] if lines else failed.status
                feedback += (f'; failed after {len(failed.ys)} evaluations; '
                             f'location={location}; error={reason}')
            else:
                prior_error = getattr(solution, 'error', None)
                explanation = evaluation_error or (str(prior_error) if prior_error else 'No generated code/seed results')
                feedback += f'; error={explanation}'
        record = dict(number=n, class_name=name, fitness=score, valid=good,
                      generation=getattr(solution, 'generation', None),
                      code=str(source.relative_to(self.folder)), feedback=feedback)
        # Save all 10 per-seed histories, not only their mean. Raw y/x values
        # are needed to replot and diagnose failures independently of the LLM.
        dump_json(self.folder / 'traces' / f'candidate_{n:03d}.json',
                  [r.to_dict() for r in runs])
        dump_json(self.folder / 'code' / f'candidate_{n:03d}.json', record)
        self.records.append(record)
        self._state(n, 'evaluated', fitness=score)
        solution.set_scores(score, feedback)
        print(f'Candidate {n:03d}: fitness={score:.5f} valid={good}', file=sys.stderr, flush=True)
        every = int(self.cfg['stage1'].get('progress_every', 5))
        if n % every == 0:
            plot_fitness(self.records, self.folder / 'fitness.png')
        return solution

    def log_individual(self, solution):
        """Count and log LLM generation failures as invalid candidates.

        Upstream LLaMEA calls this method when model generation fails before
        the ordinary evaluator is invoked. Each attempt still uses a slot in
        the assignment's maximum of 100 candidates.
        """
        if len(self.records) >= int(self.cfg['stage1']['max_candidates']):
            return
        self(solution)

    def _state(self, number, phase, **extra):
        """Input: phase. Output: atomic JSON marker for interruption diagnosis."""
        dump_json(self.folder / 'state.json', dict(candidate=number, phase=phase,
            last_evaluated=len(self.records), **extra))

    def committed(self, count, archive_folder):
        """After atomic ES archive: commit all and only archived candidates."""
        if count > len(self.records):
            raise RuntimeError('ES archive contains candidates missing from scorer records')
        # Initial empty archive is valid; do not publish uncommitted winners.
        records = self.records[:count]
        dump_json(self.folder / 'checkpoint.json', dict(
            committed=count, archive=str(archive_folder),
            best=max((r['fitness'] for r in records), default=None)))
        if records:
            winner = max((r for r in records if r['valid']),
                         key=lambda r: r['fitness'], default=None)
            if winner:
                source = self.folder / winner['code']
                (self.folder / 'best.py').write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
                dump_json(self.folder / 'best.json', dict(class_name=winner['class_name'],
                    fitness=winner['fitness'], number=winner['number'], metric=self.metric))
        self._state(count, 'committed', committed=count)


def run_target(cfg, target, resume, limit):
    """Input: YAML settings, target, resume flag, cumulative limit. Output: Stage 1 files."""
    if LLaMEA is None:
        raise RuntimeError('Install llamea before running Stage 1')
    if int(cfg['dimension']) != 6 or int(cfg['stage1']['budget']) != 60:
        raise ValueError('Assignment: dimension=6 and Stage 1 budget=60')
    es = cfg['llamea']
    parents, offspring = int(es['n_parents']), int(es['n_offspring'])
    if not (1 <= parents <= 4 and 1 <= offspring <= 4):
        raise ValueError('This runner supports 1..4 parents and 1..4 offspring')
    if not bool(es['elitism']):
        raise ValueError('Keep elitism=true for reliable small-population resume')
    if es['parent_selection'] not in ('random', 'roulette', 'tournament'):
        raise ValueError('parent_selection must be random, roulette, or tournament')
    if not (1 <= int(cfg['stage1']['seed_workers']) <= 10):
        raise ValueError('stage1.seed_workers must be 1..10')
    if float(cfg['stage1']['timeout_seconds']) <= 0:
        raise ValueError('stage1.timeout_seconds must be positive')
    if int(cfg['stage1'].get('progress_every', 5)) < 1:
        raise ValueError('stage1.progress_every must be >= 1')
    maximum = int(cfg['stage1']['max_candidates'])
    if not (parents <= limit <= maximum <= 100):
        raise ValueError(f'Cumulative limit must be {parents}..100 (at least n_parents)')
    # This calibration is independent of any generated code or evaluation seed.
    # An objective's raw scale is not supplied by the problem wrappers.
    metric = metric_for_target(cfg, target, RESULTS / 'calibration')
    # A fingerprint prevents accidentally resuming with changed problems,
    # prompts, AOCC or model configuration. The limit may INCREASE separately.
    prompt = (Path(__file__).resolve().parents[1] / cfg['prompts'][target]).read_text(encoding='utf-8')
    prompt += ("\nMandatory interface: one class with __init__(self, budget, dim) and "
        "__call__(self, func), returning (best_y, best_x). The six-dimensional "
        "minimization objective is callable ONLY as func(x); its bounds are "
        "func.bounds.lb and func.bounds.ub. Each invocation must use at most "
        "budget black-box function calls including initialization; no file, "
        "network or subprocess side effects.\n")
    signature = hashlib.sha256(json.dumps(dict(config=cfg, metric=metric, prompt=prompt),
        sort_keys=True).encode()).hexdigest()
    target_folder = RESULTS / 'stage1' / target
    if target_folder.is_symlink():
        raise RuntimeError(f'Refusing to overwrite a symlinked results folder: {target_folder}')
    folder = target_folder.resolve()
    config_file = folder / 'experiment.json'
    if resume:
        prior = json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else None
        if not prior or prior['signature'] != signature:
            raise ValueError('Cannot resume: missing run or changed configuration/prompt')
        checkpoint_path = folder / 'checkpoint.json'
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8')) if checkpoint_path.exists() else {}
        # A crash might occur AFTER the ES pickle was replaced but BEFORE its
        # JSON manifest was written. In that case the pickle is authoritative.
        archives = list(folder.rglob('llamea_config.pkl'))
        if not archives:
            raise RuntimeError('No complete LLaMEA archive found; cannot safely resume')
        saved = Path(checkpoint['archive']) if checkpoint.get('archive') else archives[0].parent
        if not (saved / 'llamea_config.pkl').is_file():
            saved = archives[0].parent
        # Unpickle ONLY your own trusted checkpoint; pickles execute Python code.
        search = AtomicLLaMEA.warm_start(str(saved))
        if search is None:
            raise RuntimeError('Cannot load a validated checkpoint; refusing to lose population history')
        committed = len(search.run_history)
        if len(search.f.records) < committed:
            raise RuntimeError('Checkpoint is missing fitness records; refusing unsafe resume')
        # The last pickle may have an incomplete callback record. Only the ES
        # run_history determines what is committed; discard any excess.
        search.f.records = search.f.records[:committed]
        if checkpoint.get('committed') != committed:
            print(f'Reconciling metadata to {committed} committed candidates', file=sys.stderr)
        # Overwrite the INCOMPLETE candidate only. Finished, committed candidates
        # and their seed traces remain untouched. LLaMEA regenerates the pending
        # candidate because no unfinished code is injected into the ES library.
        for sub, suffix in (('code', '.py'), ('code', '.json'), ('traces', '.json')):
            for file in (folder / sub).glob(f'candidate_*{suffix}'):
                try:
                    number = int(file.stem.split('_')[-1])
                except ValueError:
                    continue
                if number > committed:
                    file.unlink()
        search.f.committed(committed, saved)
        random.setstate(search._checkpoint_random_state)
        np.random.set_state(search._checkpoint_numpy_state)
        search.budget = limit
        print(f'Resuming {target}: {committed}/{limit} candidates committed')
    else:
        # Without --resume, start a fresh search and replace ONLY this target's
        # Stage 1 results. Check the LLM configuration before deleting files.
        llm = make_llm(cfg)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        dump_json(config_file, dict(config=cfg, target=target, metric=metric,
                  signature=signature, seeds=group_seeds(cfg), machine=metadata(),
                  prompt=prompt))
        seed_search(group_seeds(cfg)[0])
        scorer = CandidateScorer(cfg, target, folder, metric)
        # Construct in target folder to keep LLaMEA's native conversation logs
        # and llamea_config.pkl under this experiment, not at project root.
        previous = Path.cwd()
        os.chdir(folder)
        try:
            search = AtomicLLaMEA(f=scorer, llm=llm, task_prompt=prompt,
                budget=limit, n_parents=parents, n_offspring=offspring, max_workers=1,
                parent_selection=cfg['llamea']['parent_selection'],
                elitism=bool(cfg['llamea']['elitism']), HPO=False,
                minimization=False, log=True, experiment_name=f'BO_{target}',
                eval_timeout=10 * float(cfg['stage1']['timeout_seconds']) + 90)
        finally:
            os.chdir(previous)
    if len(search.run_history) >= limit:
        print(f'{target}: already at {len(search.run_history)} candidates; nothing to run')
        return
    previous = Path.cwd()
    try:
        os.chdir(folder)
        with warnings.catch_warnings():
            # LLaMEA runs joblib sequentially; its timeout warning is not a
            # warning about our own hard subprocess timeouts.
            warnings.filterwarnings('ignore', message='The backend class .* does not support timeout',
                                     category=UserWarning)
            search.run()
    except KeyboardInterrupt:
        print('\nInterrupted. Resume with --resume; incomplete candidate will be replaced.', file=sys.stderr)
    finally:
        os.chdir(previous)
        plot_fitness(search.f.records[:len(search.run_history)], folder / 'fitness.png')
    print(f'{target}: {len(search.run_history)} candidates committed; output {folder}')


# Stable checkpoint class identity for runs started with `python -m`.
AtomicLLaMEA.__module__ = 'runs.stage1'
CandidateScorer.__module__ = 'runs.stage1'


def main():
    """No --resume overwrites target Stage 1 results; --resume continues ES state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--target', choices=('bbob', 'realworld'), default='bbob')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Continue existing search (default: overwrite target Stage 1 results)')
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_target(cfg, args.target, args.resume,
               args.limit if args.limit is not None else int(cfg['stage1']['max_candidates']))


if __name__ == '__main__':
    main()
