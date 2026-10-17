"""Cold real API benchmark in a fresh Linux network namespace (loopback only).

Private API payloads and model logs stay in runtime/. Only aggregate diagnostics
are emitted. Each run starts a new native server and API; disk cache is retained.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import secrets
import signal
import socket
import subprocess
import sys
import time
import httpx
from scripts.launch_bonsai import command
from ml.manifest import validate_frozen_split


def rss(pid):
    try:
        fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        return int(fields['VmHWM'].split()[0]) * 1024
    except (OSError, KeyError):
        return None


def gpu_memory(pid):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    for line in result.stdout.splitlines():
        parts = [s.strip() for s in line.split(',')]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            return int(parts[1]) * 1024 * 1024
    return None


def child(args):
    subprocess.run(['ip', 'link', 'set', 'lo', 'up'], check=True, capture_output=True)
    interfaces = sorted(name for _, name in socket.if_nameindex())
    routes = Path('/proc/net/route').read_text().splitlines()[1:]
    if interfaces != ['lo'] or routes:
        raise RuntimeError('Benchmark requires isolated loopback-only networking')
    root = Path(args.output).parent / f'cold-run-{args.child_index}-{time.time_ns()}'; root.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.manifest).read_text())['videos']
    video = Path(args.manifest).parent / rows[args.child_index]['path']
    start = time.monotonic(); processes = []; logs = []
    report = {'network_interfaces': interfaces, 'external_routes': len(routes),
              'duration_sec': rows[args.child_index]['duration_sec'], 'status': 'failed'}
    client = httpx.Client(timeout=10, trust_env=False, follow_redirects=False)
    try:
        log = (root / 'native.log').open('w'); logs.append(log)
        native = subprocess.Popen(command(args.config, args.binary), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(native)
        def wait_ready(url, process):
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Local subprocess failed during startup')
                try:
                    if client.get(url).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(.25)
            raise TimeoutError('Startup exceeded 60 seconds')
        endpoint = json.loads(Path(args.config).read_text())['endpoint']
        wait_ready(endpoint + '/health', native)
        report['native_cold_load_sec'] = time.monotonic() - start
        key = secrets.token_urlsafe(32)
        env = dict(os.environ, FORTE_API_KEY=key)
        log = (root / 'api.log').open('w'); logs.append(log)
        api = subprocess.Popen([sys.executable, '-m', 'scripts.serve_bonsai', '--config', args.config,
                                '--port', '18081', '--runtime', str(root / 'api')], env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(api); wait_ready('http://127.0.0.1:18081/health', api)
        request_start = time.monotonic()
        headers = {'X-API-Key': key}
        with video.open('rb') as stream:
            response = client.post('http://127.0.0.1:18081/api/v1/analyze', headers=headers,
                                   files={'file': ('benchmark.mov', stream, 'video/quicktime')}, timeout=30)
        response.raise_for_status(); job = response.json()['job_id']
        peaks = []
        while time.monotonic() - start < args.timeout:
            response = client.get('http://127.0.0.1:18081/api/v1/jobs/' + job, headers=headers)
            response.raise_for_status(); result = response.json()
            value = gpu_memory(native.pid)
            if value is not None: peaks.append(value)
            if result['status'] != 'processing':
                break
            time.sleep(.5)
        else:
            raise TimeoutError('Full cold API run exceeded configured time limit')
        (root / 'private-result.json').write_text(json.dumps(result))
        report.update(status=result['status'], api_cycle_sec=time.monotonic() - request_start,
                      cold_total_sec=time.monotonic() - start,
                      peak_native_process_vram_bytes=max(peaks) if peaks else None,
                      peak_native_rss_bytes=rss(native.pid), peak_api_rss_bytes=rss(api.pid))
        if result['status'] == 'done':
            report.update(events=len(result['violations']), verdict=result['verdict'],
                          evidence_counts=[len(v['evidence_frames']) for v in result['violations']])
        else:
            from scripts.evaluate import failure_reason
            report['failure_reason'] = failure_reason(RuntimeError(result.get('error', 'unknown')))
    except Exception as error:
        from scripts.evaluate import failure_reason
        report.update(failure_reason=failure_reason(error), cold_total_sec=time.monotonic() - start)
    finally:
        client.close()
        for process in reversed(processes):
            if process.poll() is None: os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        for log in logs: log.close()
    report['max_child_rss_bytes_not_sum'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/real/manifest.draft.json')
    p.add_argument('--split-file', default='runtime/weak-baseline/split.freeze.json')
    p.add_argument('--config', default='configs/bonsai.json')
    p.add_argument('--binary', default='runtime/bonsai-setup/build/bin/llama-server')
    p.add_argument('--output', default='runtime/bonsai-cold/report.json')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--child-index', type=int)
    a = p.parse_args()
    if a.child_index is not None:
        child(a); return
    rows = json.loads(Path(a.manifest).read_text())['videos']
    frozen = json.loads(Path(a.split_file).read_text())
    validate_frozen_split(rows, frozen)
    index = max((i for i, split in enumerate(frozen['splits']) if split == a.split),
                key=lambda i: rows[i]['duration_sec'])
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); reports = []
    for _ in range(a.runs):
        argv = ['unshare', '--user', '--map-root-user', '--net', sys.executable,
                '-m', 'scripts.benchmark_bonsai', '--manifest', a.manifest,
                '--config', a.config, '--binary', a.binary, '--output', a.output,
                '--timeout', str(a.timeout), '--child-index', str(index)]
        process = subprocess.run(argv, capture_output=True, text=True, timeout=a.timeout + 150)
        if process.returncode:
            (out.parent / 'private-benchmark-error.log').write_text(process.stderr)
            raise RuntimeError('Isolated benchmark failed; private stderr was not printed')
        report = json.loads(process.stdout); reports.append(report)
        result = {'split': a.split, 'runs': reports, 'disk_cache_flushed': False,
                  'note': 'Fresh native/API processes each run; private outputs remain local.'}
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(report), flush=True)
        if report['status'] != 'done': break


if __name__ == '__main__':
    main()
