"""Bounded startup model probe using synthetic pixels, never uploaded footage."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

from app.processes import run_child

MESSAGES = {
    'not_checked': 'Model startup check has not completed',
    'model_missing': 'Model files are missing; complete setup and restart',
    'probe_timeout': 'Model startup check exceeded its time budget',
    'dependencies_missing': 'Model dependencies are missing in this Python environment',
    'model_invalid': 'Model, rules or inference response failed the startup check',
    'probe_failed': 'Model could not load or run; check local installation and available memory',
    'restart_required': 'Model or rules changed; restart to check the new configuration',
}


def fingerprint(settings):
    paths = [settings.rules]
    if settings.model.is_dir():
        paths += sorted(p for p in settings.model.rglob('*')
                        if p.is_file() and p.suffix in {'.json', '.safetensors', '.gguf', '.npz'})
    else:
        paths.append(settings.model)
    return tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def probe_model(model, engine, rules):
    import numpy as np
    from ml.engine import load_rules, reference_scores
    from ml.video import Frame

    enabled = load_rules(rules, engine == 'demo')
    if not enabled:
        raise ValueError('No enabled rules')
    frames = [Frame(t, np.zeros((64, 64, 3), dtype=np.uint8)) for t in (0., 2., 3.)]
    if engine in {'reference', 'demo'}:
        labels, scores, threshold = reference_scores(frames, model, engine == 'demo')
        if ('normal' not in labels or len(labels) < 2 or len(set(labels)) != len(labels)
                or any(label != 'normal' and label not in enabled for label in labels)
                or scores.shape != (len(frames), len(labels))
                or not np.isfinite(scores).all() or not 0 < threshold < 1):
            raise ValueError('Invalid model shape, values or class-to-rule mapping')
    elif engine in {'qwen', 'bonsai'}:
        from ml.qwen import QwenSession, parse_events, qwen_prompt
        if engine == 'qwen':
            session = QwenSession(model)
        else:
            from ml.bonsai import BonsaiSession
            session = BonsaiSession(model)
        try:
            parse_events(session.infer(frames, qwen_prompt(enabled)), enabled, 0, 3)
        finally:
            if hasattr(session, 'close'):
                session.close()
    else:
        raise ValueError('Unsupported engine')


class ModelReadiness:
    def __init__(self, settings):
        self.settings = settings
        self.checked_fingerprint = None
        self.report = {'ready': False, 'code': 'not_checked', 'message': MESSAGES['not_checked']}

    def check(self):
        started = time.monotonic()
        output = self.settings.runtime / ('.readiness-' + uuid.uuid4().hex + '.json')
        try:
            if not self.settings.model.exists():
                raise FileNotFoundError()
            before = fingerprint(self.settings)
            command = ['--model', str(self.settings.model.resolve()),
                       '--engine', self.settings.engine, '--rules', str(self.settings.rules.resolve()),
                       '--output', str(output.resolve())]
            code = run_child('probe', command, self.settings.readiness_timeout)
            value = json.loads(output.read_text(encoding='utf-8')) if code == 0 else {}
            code = value.get('code', 'probe_failed')
            if value.get('ready') is True:
                if fingerprint(self.settings) != before:
                    code = 'restart_required'
                else:
                    self.checked_fingerprint = before
                    self.report = {'ready': True, 'code': None, 'message': 'Model loaded and synthetic inference passed'}
                    return
            self.report = {'ready': False, 'code': code, 'message': MESSAGES.get(code, MESSAGES['probe_failed'])}
        except FileNotFoundError:
            self.report = {'ready': False, 'code': 'model_missing', 'message': MESSAGES['model_missing']}
        except subprocess.TimeoutExpired:
            self.report = {'ready': False, 'code': 'probe_timeout', 'message': MESSAGES['probe_timeout']}
        except (OSError, ValueError, KeyError):
            self.report = {'ready': False, 'code': 'probe_failed', 'message': MESSAGES['probe_failed']}
        finally:
            self.report['check_sec'] = round(time.monotonic() - started, 3)
            self.report['checked_at'] = time.time()
            try:
                output.unlink(missing_ok=True)
            except OSError:
                pass

    def snapshot(self):
        value = dict(self.report)
        if value['ready']:
            try:
                changed = fingerprint(self.settings) != self.checked_fingerprint
            except OSError:
                changed = True
            if changed:
                value.update(ready=False, code='restart_required', message=MESSAGES['restart_required'])
        return value


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'engine', 'rules', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        probe_model(args.model, args.engine, args.rules)
        result = {'ready': True}
    except ImportError:
        result = {'ready': False, 'code': 'dependencies_missing'}
    except (ValueError, KeyError, TypeError):
        result = {'ready': False, 'code': 'model_invalid'}
    except Exception:
        result = {'ready': False, 'code': 'probe_failed'}
    Path(args.output).write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main()
