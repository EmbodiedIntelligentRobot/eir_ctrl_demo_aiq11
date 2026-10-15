"""Opt-in T10 MuJoCo A/B capture. No hardware; reports observations, not a jitter SLA."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import yaml


def stats(values):
    values = sorted(values)
    def percentile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {'n': len(values), 'min': values[0], 'mean': sum(values)/len(values),
            'p50': percentile(.5), 'p99': percentile(.99), 'max': values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['onnx', 'libtorch'], default='libtorch')
    parser.add_argument('--modes', type=int, nargs='+', choices=[1, 2, 28, 41], default=[28, 41])
    parser.add_argument('--warmup', type=float, default=10)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0 or args.seconds <= 0 or args.repeats < 1:
        parser.error('Invalid measurement duration/repeats')
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.config.resolve()
    config = yaml.safe_load(source.read_text())
    if config['runtime']['backend'] != 'simulation':
        parser.error('Only simulation configuration is accepted')
    for key in ['robot', 'motion_stream', 'flight_recorder']:
        if config.get(key): config[key] = str((source.parent / config[key]).resolve())
    config['policies'] = {k: str((source.parent / v).resolve()) for k, v in config['policies'].items()}
    config['runtime']['model'] = str((source.parent / config['runtime']['model']).resolve())
    config['inference']['backend'] = args.backend
    config['runtime']['headless'] = True
    config['runtime']['suspend'] = True
    config['runtime']['release_after_ready'] = False
    config['remote']['gui'] = False
    config['remote']['source'] = 'external'
    config_path = args.output.resolve() / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    environment = dict(os.environ, ROS_LOCALHOST_ONLY='1', ROS_DOMAIN_ID='186')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = {'schema': 1, 'backend': args.backend, 'warmup_seconds': args.warmup,
              'capture_seconds': args.seconds, 'repeats': args.repeats,
              'host': platform.node(), 'platform': platform.platform(),
              'cpu_count': os.cpu_count(), 'domain': 186, 'suspended': True,
              'expected_period_us': 1e6/config['timing']['control_hz'],
              'executable_sha256': sha(args.executable), 'source_config_sha256': sha(source),
              'top_level_mjcf_sha256': sha(config['runtime']['model']), 'runs': []}
    # Includes all model assets, not only the top-level XML with include directives.
    model_root = Path(config['runtime']['model']).parents[1]
    result['model_assets_sha256'] = {str(p.relative_to(model_root)): sha(p)
        for p in sorted(model_root.rglob('*')) if p.is_file()}
    profile_assets = set()
    for value in config['policies'].values():
        profile_assets.update(p for p in Path(value).parent.rglob('*') if p.is_file())
    result['policy_assets_sha256'] = {str(p): sha(p) for p in sorted(profile_assets)}
    for repeat in range(args.repeats):
        for mode in args.modes:
            for toggle in (['off', 'on'] if repeat % 2 == 0 else ['on', 'off']):
                name = f'{args.backend}-mode{mode}-r{repeat+1}-{toggle}'
                path = args.output / f'{name}.csv'
                run = {'name': name, 'mode': mode, 'debug': toggle,
                       'started_unix': time.time(), 'loadavg_before': os.getloadavg()}
                command = [str(args.executable.resolve()), str(config_path), toggle,
                           str(mode), str(args.warmup), str(args.seconds), str(path.resolve()), args.backend]
                with (args.output / f'{name}.log').open('w') as log:
                    process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=args.warmup + args.seconds + 60)
                run['exit_code'] = process.returncode
                run['elapsed_seconds'] = time.time() - run['started_unix']
                result['runs'].append(run)
                if process.returncode:
                    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
                    raise RuntimeError(f'{name} failed; see log and partial CSV')
                lines = (args.output/f'{name}.log').read_text().splitlines()
                run['diagnostics'] = json.loads(next(line.removeprefix('T10_RESULT ') for line in lines if line.startswith('T10_RESULT ')))
                with path.open() as stream:
                    samples = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
                period = result['expected_period_us']
                run['interval_us'] = stats([s['interval_ns']/1000 for s in samples])
                run['execution_us'] = stats([s['execution_ns']/1000 for s in samples])
                run['absolute_jitter_us'] = stats([abs(s['interval_ns']/1000-period) for s in samples])
                run['interval_gt_3ms'] = sum(s['interval_ns'] > 3000000 for s in samples)
                run['execution_gt_period'] = sum(s['execution_ns']/1000 > period for s in samples)
                run['overrun_events'] = sum(s['skipped_ticks'] > 0 for s in samples)
                run['skipped_ticks'] = sum(s['skipped_ticks'] for s in samples)
                run['fault_samples'] = sum(s['fault'] != 0 for s in samples)
                with path.open('rb') as stream, gzip.open(str(path)+'.gz', 'wb') as archive:
                    shutil.copyfileobj(stream, archive)
                path.unlink()
                run['csv_gz_sha256'] = sha(str(path)+'.gz')
                (args.output/'summary.json').write_text(json.dumps(result, indent=2))
                print(json.dumps(run), flush=True)
    print(f'T10 capture complete: {args.output}/summary.json', flush=True)


if __name__ == '__main__':
    main()
