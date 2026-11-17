"""Generate BO algorithms with installed LLaMEA and resumable small (mu+lambda) search.

CLI:
  python -m runs.stage1 --target bbob --limit 20
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
# entry point is invoked using `python -m runs.stage1`.
if __name__ == '__main__':
    sys.modules['runs.stage1'] = sys.modules[__name__]

from src.problems import from_config, metric_for_target
from src.performance import Algorithm, evaluate_many
from src.utils import (DEFAULT_CONFIG, RESULTS, dump_json, group_seeds,
                       load_config, metadata, plot_fitness, seed_search)

# LLaMEA manages generation, operators and selection.
# The subclass bounds offspring batches and checkpoints ES state.
try:
    from llamea import LLaMEA, Operator
except ImportError:
    LLaMEA = Operator = None


# LLaMEA operator definitions
# Prompt content and selection weights live outside the execution script.
# Crossover requires two parents; all other operators mutate one parent.
OPERATOR_PARENTS = {
    'refine': 1,
    'surrogate': 1,
    'acquisition': 1,
    'trust_region': 1,
    'hybrid': 1,
    'init': 1,
    'crossover': 2,
}


def build_operators(cfg):
    """Load the selected LLaMEA operator portfolio from YAML and text files."""
    es = cfg['llamea']
    variant = es.get('operators', 'baseline')
    if variant == 'baseline':
        return []  # Use the library default mutation operator.
    if variant != 'novelty':
        raise ValueError('llamea.operators must be baseline or novelty')

    directory = (Path(__file__).resolve().parents[1] / cfg['prompts']['operator_dir'])
    common = (directory / 'common.txt').read_text(encoding='utf-8').strip()
    weights = es['operator_weights']
    unknown = set(weights) - set(OPERATOR_PARENTS)
    if unknown:
        raise ValueError(f'Unknown LLaMEA operators: {sorted(unknown)}')

    operators = []
    for name, parents in OPERATOR_PARENTS.items():
        weight = float(weights.get(name, 0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f'Operator {name} weight must be a finite nonnegative number')
        if weight == 0 or parents > int(es['n_parents']):
            continue
        instruction = (directory / f'{name}.txt').read_text(encoding='utf-8').strip()
        operators.append(Operator(
            name=name,
            prompt=instruction + '\n\n' + common,
            weight=weight,
            number_of_parents=parents,
        ))
    if not operators:
        raise ValueError('Novelty mode needs at least one eligible positive-weight operator')
    return operators


def stage1_output_folder(target, tag=None, rep=0):
    """Route untagged runs to stage1 and tagged repetitions to stage1_runs."""
    if rep < 0:
        raise ValueError('--rep must be nonnegative')
    if tag is None:
        if rep != 0:
            raise ValueError('--rep requires --tag')
        return RESULTS / 'stage1' / target
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', tag):
        raise ValueError('--tag must be 1-64 letters, digits, underscores or hyphens, starting with a letter or digit')
    return RESULTS / 'stage1_runs' / f'{tag}_r{rep}' / target


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
        # Persist the evolutionary RNG states for resume.
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
        # Commit the candidate after archiving the evolutionary state.
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
        # Invalid candidates receive zero fitness.
        score = float(np.mean([r.aocc for r in runs])) if good else 0.0
        feedback = f'Fitness (mean AOCC across 10 runs): {score:.6f}; valid={good}'
        if good:
            final = float(np.mean([r.best for r in runs]))
            initial = float(np.mean([min(r.ys[:min(8, len(r.ys))]) for r in runs]))
            feedback += f'; mean initial best={initial:.6g}; mean final best={final:.6g}'
        if good and self.cfg['llamea'].get('feedback', 'basic') == 'rich':
            # Include convergence checkpoints and variation across seeds.
            marks = [i for i in (10, 20, 40, 60) if all(len(r.ys) >= i for r in runs)]
            at = ', '.join(f'{i}:{np.mean([min(r.ys[:i]) for r in runs]):.4g}' for i in marks)
            aoccs = [r.aocc for r in runs]
            stalled = sum(1 for r in runs if len(r.ys) > 30 and min(r.ys[:30]) == min(r.ys))
            feedback += (f'; mean best-so-far at eval [{at}]; per-seed AOCC min/max='
                         f'{min(aoccs):.3f}/{max(aoccs):.3f}; seeds with no improvement after eval 30: {stalled}/10')
        if not good:
            # Return the failing code location and final exception in feedback.
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
        # Retain the complete seeded traces for plotting and diagnostics.
        dump_json(self.folder / 'traces' / f'candidate_{n:03d}.json',
                  [r.to_dict() for r in runs])
        dump_json(self.folder / 'code' / f'candidate_{n:03d}.json', record)
        self.records.append(record)
        self._state(n, 'evaluated', fitness=score)
        solution.set_scores(score, feedback)
        print(f'Candidate {n:03d}: fitness={score:.5f} valid={good}', file=sys.stderr, flush=True)
        every = int(self.cfg['stage1'].get('progress_every', 5))
        if n % every == 0:
            plot_fitness(self.records, self.folder / 'fitness.png',
                         target=self.target, normalized=self.metric.get('scale') is not None)
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


def run_target(cfg, target, resume, limit, tag=None, rep=0):
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
    # Validate output routing before calibration or any file writes.
    target_folder = stage1_output_folder(target, tag, rep)
    # Calibrate the objective scale separately from seeded optimizer runs.
    metric = metric_for_target(cfg, target, RESULTS / 'calibration')
    # Bind checkpoints to the effective configuration, prompts and metric.
    # The cumulative candidate limit may increase on resume.
    prompt = (Path(__file__).resolve().parents[1] / cfg['prompts'][target]).read_text(encoding='utf-8')
    prompt += ("\nMandatory interface: one class with __init__(self, budget, dim) and "
        "__call__(self, func), returning (best_y, best_x). The six-dimensional "
        "minimization objective is callable ONLY as func(x); its bounds are "
        "func.bounds.lb and func.bounds.ub. Each invocation must use at most "
        "budget black-box function calls including initialization; no file, "
        "network or subprocess side effects.\n")
    operators = build_operators(cfg)
    signature = hashlib.sha256(json.dumps(dict(config=cfg, metric=metric, prompt=prompt,
        operators=[o.prompt for o in operators], rep=rep),
        sort_keys=True).encode()).hexdigest()
    # Tagged search outputs are isolated from the default Stage 2 winners.
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
        # Prefer the complete ES pickle if the manifest is out of date.
        archives = list(folder.rglob('llamea_config.pkl'))
        if not archives:
            raise RuntimeError('No complete LLaMEA archive found; cannot safely resume')
        saved = Path(checkpoint['archive']) if checkpoint.get('archive') else archives[0].parent
        if not (saved / 'llamea_config.pkl').is_file():
            saved = archives[0].parent
        # Only load trusted local checkpoints; pickle can execute code.
        search = AtomicLLaMEA.warm_start(str(saved))
        if search is None:
            raise RuntimeError('Cannot load a validated checkpoint; refusing to lose population history')
        committed = len(search.run_history)
        if len(search.f.records) < committed:
            raise RuntimeError('Checkpoint is missing fitness records; refusing unsafe resume')
        # Commit only records present in the saved evolutionary history.
        search.f.records = search.f.records[:committed]
        if checkpoint.get('committed') != committed:
            print(f'Reconciling metadata to {committed} committed candidates', file=sys.stderr)
        # Discard uncommitted candidate files; retain completed traces.
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
        # A fresh search replaces only its target directory after LLM setup.
        llm = make_llm(cfg)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        dump_json(config_file, dict(config=cfg, target=target, metric=metric,
                  signature=signature, seeds=group_seeds(cfg), machine=metadata(),
                  prompt=prompt))
        seed_search(group_seeds(cfg)[0] + rep)  # rep changes only the ES/LLM-side randomness
        scorer = CandidateScorer(cfg, target, folder, metric)
        # Keep LLaMEA logs and checkpoints in the selected experiment folder.
        previous = Path.cwd()
        os.chdir(folder)
        try:
            search = AtomicLLaMEA(f=scorer, llm=llm, task_prompt=prompt,
                budget=limit, n_parents=parents, n_offspring=offspring, max_workers=1,
                parent_selection=cfg['llamea']['parent_selection'],
                tournament_size=int(es.get('tournament_size', 3)), operators=operators,
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
            # Ignore joblib's sequential-backend notice; worker timeouts remain enforced.
            warnings.filterwarnings('ignore', message='The backend class .* does not support timeout',
                                     category=UserWarning)
            search.run()
    except KeyboardInterrupt:
        print('\nInterrupted. Resume with --resume; incomplete candidate will be replaced.', file=sys.stderr)
    finally:
        os.chdir(previous)
        plot_fitness(search.f.records[:len(search.run_history)],
                     folder / 'fitness.png', target=target,
                     normalized=metric.get('scale') is not None)
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
    parser.add_argument('--tag', default=None, help='Experiment name; output goes to stage1_runs/<tag>_r<rep>/')
    parser.add_argument('--rep', type=int, default=None, help='Repeat index; requires --tag (default 0 for tagged runs)')
    parser.add_argument('--parents', type=int, default=None)
    parser.add_argument('--offspring', type=int, default=None)
    parser.add_argument('--selection', choices=('random', 'roulette', 'tournament'), default=None)
    parser.add_argument('--tournament-size', type=int, default=None)
    parser.add_argument('--operators', choices=('baseline', 'novelty'), default=None)
    parser.add_argument('--feedback', choices=('basic', 'rich'), default=None)
    args = parser.parse_args()
    # Repetition indices are only meaningful for tagged searches.
    if args.tag is None and args.rep is not None:
        parser.error('--rep requires --tag')
    cfg = load_config(args.config)
    for key, value in (('n_parents', args.parents), ('n_offspring', args.offspring),
                       ('parent_selection', args.selection), ('tournament_size', args.tournament_size),
                       ('operators', args.operators), ('feedback', args.feedback)):
        if value is not None:
            cfg['llamea'][key] = value
    run_target(cfg, args.target, args.resume,
               args.limit if args.limit is not None else int(cfg['stage1']['max_candidates']),
               tag=args.tag, rep=0 if args.rep is None else args.rep)


if __name__ == '__main__':
    main()