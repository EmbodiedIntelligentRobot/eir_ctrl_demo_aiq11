"""Explicit, headless AIQ recorder acceptance; never connects to hardware."""
import argparse
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import yaml

WORKSPACE = Path(__file__).resolve().parents[3]
APP = WORKSPACE / 'apps/eir_ctrl_demo_aiq11'


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def profiles(directory, backends, model):
    directory.mkdir()
    for name, source in [('pro', 'pro_special_mimic.yaml'),
                         ('sport', 'sport_special_mimic.yaml'),
                         ('perception', 'pro_perception.yaml'),
                         ('perception-split', 'pro_perception.yaml'),
                         ('motion', 'pro_motion.yaml')]:
        file = APP / 'config' / source
        for backend in backends:
            config = yaml.safe_load(file.read_text())
            for key in ['robot', 'flight_recorder', 'motion_stream']:
                if isinstance(config.get(key), str):
                    config[key] = str((file.parent / config[key]).resolve())
            config['policies'] = {mode: str((file.parent / path).resolve())
                                  for mode, path in config['policies'].items()}
            config['runtime']['model'] = str(model) if name != 'sport' else str(
                (file.parent / config['runtime']['model']).resolve())
            config['inference']['backend'] = backend
            if name.startswith('perception'):
                config['inference']['perception_execution'] = (
                    'split' if name.endswith('split') else 'combined')
            if name.endswith('split'):
                config['inference']['encoder_hz'] = 50.0
            (directory / f'{name}-{backend}.yaml').write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False))


def run_matrix(executable, output, backends, model):
    output.mkdir()  # Refuse to overwrite earlier evidence.
    configs = output / 'profiles'
    profiles(configs, backends, model)
    results = []
    for profile in ['pro', 'sport', 'perception', 'perception-split', 'motion']:
        scopes = ['motion'] if profile == 'motion' else (
            ['suspended', 'ground'] if profile in ['pro', 'sport'] else ['suspended', 'depth_fault'])
        for backend in backends:
            for scope in scopes:
                name = f'{profile}-{backend}-{scope}'
                directory = output / name
                log = output / (name + '.log')
                started = time.monotonic()
                with log.open('w') as stream:
                    code = subprocess.run([str(executable), str(configs / f'{profile}-{backend}.yaml'),
                                           backend, str(directory), scope], stdout=stream,
                                          stderr=subprocess.STDOUT).returncode
                result = {'case': name, 'profile': profile, 'backend': backend, 'scope': scope,
                          'exit_code': code, 'wall_seconds': round(time.monotonic() - started, 3),
                          'output': str(directory), 'log': str(log)}
                if (directory / 'acceptance.json').exists():
                    result.update(json.loads((directory / 'acceptance.json').read_text()))
                results.append(result)
                save_json(output / 'matrix.json', results)
                print(name, 'PASS' if code == 0 else 'FAIL', flush=True)
    return all(case['exit_code'] == 0 for case in results)


def check_records(output):
    spec = importlib.util.spec_from_file_location(
        'flight_decode', WORKSPACE / 'src/eir_loco_runtime/tools/解码飞行记录.py')
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    checks = []
    for case in json.loads((output / 'matrix.json').read_text()):
        assert case['exit_code'] == 0, (case['case'], 'simulation failed; inspect its log')
        config = yaml.safe_load((output / 'profiles' / f"{case['profile']}-{case['backend']}.yaml").read_text())
        policies = {mode: yaml.safe_load(Path(path).read_text()) for mode, path in config['policies'].items()}
        runs = list(Path(case['output']).glob('eirflight_*'))
        assert len(runs) == 1, case['case']
        run = runs[0]
        metadata = json.loads((run / 'metadata.json').read_text())
        statistics = json.loads((run / 'statistics.json').read_text())
        issues, causes, counts, modes = [], [], {}, set()
        selected = {}
        actions = 0
        for file in [run / 'events.bin', *run.glob('incident_*.bin'), *run.glob('snapshot_*.bin')]:
            count = 0
            for row in decoder.records(file, issues):
                count += 1
                if row['source'] != 'application':
                    continue
                mode = row.get('app_codes.0')
                if mode is not None:
                    modes.add(mode)
                if 'failure_cause.0' in row:
                    causes.append([row['failure_cause.0'], row['failure_cause.1']])
                # Preserve actual fault/request examples and one action per session/mode.
                key = (row.get('session', row.get('failure_session', 0)), mode,
                       row.get('event'), row.get('mode_request.1'), row.get('motion_generation'),
                       row.get('policy_state.10'), row.get('policy_state.11'))
                selected.setdefault(key, row)
                if row.get('policy_state.1') != 1:
                    continue
                actions += 1
                n = row['policy_state.0']
                assert n in [23, 29], (case['case'], n)
                assert row['target_session'] == row['session'] or row['app_codes.1'] != 0
                assert row['target_source_stamp_ns'] <= row['target_stamp_ns']
                assert row['feedback_stamp_ns'] <= row['source_ns']
                assert row['io_feedback_sequence'] >= row['feedback_sequence']
                policy = policies.get(mode)
                for i in range(n):
                    assert f'policy_raw_action.{i}' in row and f'policy_applied_action.{i}' in row
                    if policy:
                        joint = row[f'policy_joint_indices.{i}']
                        target = policy['defaults'][i] + policy['scales'][i] * row[f'policy_applied_action.{i}']
                        assert math.isclose(row[f'policy_target_q.{joint}'], target, abs_tol=1e-9)
                if case['robot'] == 'sport':
                    assert not any(key.startswith('mechanism_3_') for key in row)
            counts[file.name] = count
        assert not issues and actions > 0, (case['case'], issues, actions)
        assert statistics['write_failures'] == 0 and statistics['dropped'] == 0, case['case']
        assert any(':' + case['backend'] + ':' in value for value in metadata['controllers'])
        if case['scope'] == 'depth_fault':
            assert [4, 2] in causes and [1, 9] not in causes, (case['case'], causes)
        else:
            assert not causes, (case['case'], causes)
        with gzip.open(Path(case['output']) / 'selected-application-records.jsonl.gz', 'wt') as stream:
            for row in selected.values():
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        checks.append({'case': case['case'], 'counts': counts, 'modes': sorted(modes),
                       'policy_records': actions, 'failure_causes': causes,
                       'write_failures': statistics['write_failures'], 'drops': statistics['dropped'],
                       'metadata_config_digest': metadata['config_digest']})
        save_json(output / 'decoded-check.json', checks)
        print(case['case'], 'decoded and checked', actions, 'policy records', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', nargs='+', choices=['onnx', 'libtorch'], default=['onnx', 'libtorch'])
    parser.add_argument('--decode-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.decode_only:
        if not args.executable or not args.model:
            parser.error('--executable and --model are required to run simulations')
        os.environ['AIQ11_PRO_MODEL'] = str(args.model.resolve())
        if not run_matrix(args.executable.resolve(), output, args.backend, args.model.resolve()):
            raise SystemExit('Acceptance failed; earlier evidence retained for diagnosis')
    check_records(output)
