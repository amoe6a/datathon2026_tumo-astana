"""Start the separately built pinned Bonsai server on loopback, without downloads."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from ml.bonsai import checked_endpoint


def command(config, binary):
    settings = json.loads(Path(config).read_text())
    endpoint = urlsplit(checked_endpoint(settings['endpoint']))
    binary = Path(binary).resolve()
    weights = Path(settings.get('weights', 'models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf')).resolve()
    projector = Path(settings.get('projector', 'models/bonsai/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf')).resolve()
    if not all(p.is_file() for p in (binary, weights, projector)):
        raise ValueError('Native binary, weights and projector must already exist locally')
    return [str(binary), '--model', str(weights), '--mmproj', str(projector),
            '--host', endpoint.hostname, '--port', str(endpoint.port or 80),
            '--alias', settings['model_alias'], '--ctx-size', str(settings['context_tokens']),
            '--parallel', str(settings['parallel']), '--gpu-layers', str(settings['gpu_layers']),
            '--batch-size', str(settings['batch_size']), '--ubatch-size', str(settings['ubatch_size']),
            '--threads', str(settings.get('cpu_threads', 12)),
            '--threads-batch', str(settings.get('cpu_threads', 12)), '--flash-attn', 'on',
            '--no-mmproj-offload', '--image-min-tokens', str(settings['image_min_tokens']),
            '--image-max-tokens', str(settings['image_max_tokens']),
            '--reasoning-budget', str(settings['reasoning_budget']), '--offline', '--no-webui']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/bonsai.json')
    parser.add_argument('--binary', default='runtime/bonsai-setup/build/bin/llama-server')
    args = parser.parse_args()
    argv = command(args.config, args.binary)
    os.execv(argv[0], argv)


if __name__ == '__main__':
    main()
