"""Owned child processes with deadlines and parent-death supervision."""
import os
import subprocess
import sys

from app.config import ROOT


def start_child(mode, arguments):
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_HUB_DISABLE_TELEMETRY': '1'}
    env.pop('FORTE_API_KEY', None)
    return subprocess.Popen([sys.executable, '-m', 'app.worker_host', mode, *arguments],
                            cwd=ROOT, env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            close_fds=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def stop_child(process):
    try:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
    finally:
        if process.stdin is not None:
            process.stdin.close()


def run_child(mode, arguments, timeout):
    process = start_child(mode, arguments)
    try:
        return process.wait(timeout=timeout)
    finally:
        stop_child(process)
