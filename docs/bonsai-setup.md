# Локальный Bonsai: воспроизведение эксперимента

Это экспериментальный кандидат, а не готовая к соревнованию модель. Результаты и ограничения находятся в [ML handoff](ml-handoff.md). Команды выполняются из корня ветки `codex/ml-local-evaluation`.

## Артефакты и сборка

Требуются Python-окружение с базовыми requirements, CMake, CUDA toolkit и подходящий C/C++ compiler. На проверенном компьютере использовались CUDA 13.4, GCC/G++ 14 и архитектура 89 (RTX 4060 Laptop). Системные пакеты эти команды не устанавливают.

Скачивание — только подготовительный этап. Используйте отдельное GPU/setup-окружение с huggingface-hub; его установка не нужна для последующего нативного inference. Скрипт фиксирует ревизию издателя и сверяет оба SHA-256:

```bash
.venv-inference/bin/python -m scripts.download_bonsai
# Если файлы уже находятся в models/bonsai, сеть не нужна:
.venv/bin/python -m scripts.download_bonsai --verify-only
```

Форк runtime: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), release `prism-b10777-e3de715`, commit `e3de715222b2667c54a267441159a5729f11155e`. Обычный сторонний llama.cpp не подставлялся вместо этого форка.

```bash
mkdir -p runtime/bonsai-setup
# Только для нового checkout; не перезаписывайте имеющуюся папку source.
git clone --depth 1 --branch prism-b10777-e3de715 https://github.com/PrismML-Eng/llama.cpp.git runtime/bonsai-setup/source
git -C runtime/bonsai-setup/source checkout --detach e3de715222b2667c54a267441159a5729f11155e
git -C runtime/bonsai-setup/source rev-parse HEAD
# Перед сборкой вывод должен совпасть с закреплённым commit выше.
cmake -S runtime/bonsai-setup/source -B runtime/bonsai-setup/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-14 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_IS_DEV=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build runtime/bonsai-setup/build --target llama-server -j 6
```

В текущем эксперименте из-за срывов git/archive исходники были получены по файлам и проверены по git blob SHA закреплённого дерева. Полные локальные `tree.json` и `source-manifest.json` сохранены в runtime/. Успешная сборка проверена на этой машине; команды нового git clone выше не означают проверки доступности сети на другой машине.

## Запуск и измерения

```bash
# Терминал 1: нативный runtime, только loopback; веса уже локальные
.venv/bin/python -m scripts.launch_bonsai --config configs/bonsai.json

# Терминал 2: экспериментальная API-точка
# FORTE_API_KEY задаётся локально, не публикуется и не записывается в Git.
.venv/bin/python -m scripts.serve_bonsai --config configs/bonsai.json
```

API сохраняет прежний контракт: POST /api/v1/analyze, GET /api/v1/jobs/{job_id}, X-API-Key. Поле FORTE_ENGINE=bonsai в основном Settings.from_env ещё не добавлено; используется именно экспериментальный entrypoint.

Для cold benchmark сначала остановите собственный обычный native server, чтобы не держать две копии весов в VRAM. Скрипт сам запускает и завершает отдельные native/API-процессы, генерирует временный ключ без вывода в терминал и изолирует сеть Linux namespace. Приватные API-результаты и логи остаются в игнорируемой runtime/.

```bash
.venv/bin/python -m scripts.benchmark_bonsai --runs 1 --split train --output runtime/bonsai-cold-train/report.json
.venv/bin/python -m scripts.benchmark_bonsai --runs 3 --split val --output runtime/bonsai-cold-val/report.json
```

Это новые процессы при каждом запуске, но файловый кеш ОС не очищается. VRAM — измеренная память native-процесса через nvidia-smi, RAM — VmHWM отдельных native/API-процессов; их нельзя выдавать за суммарный пик всей системы. Namespace проверяется через реальные сетевые интерфейсы и отсутствие внешних маршрутов; настройки сети хоста не меняются.

Псевдоразметка не является независимой ground truth. Она запускается локально, сохраняет подробности только в data/ и требует отдельной проверки людьми:

```bash
.venv/bin/python -m scripts.auto_annotate data/real/manifest.draft.json --engine bonsai --model configs/bonsai.json --split train --output data/real/pseudo-bonsai
.venv/bin/python -m scripts.auto_annotate data/real/manifest.draft.json --engine bonsai --model configs/bonsai.json --split val --output data/real/pseudo-bonsai
.venv/bin/python -m scripts.evaluate_source_labels data/real/manifest.draft.json data/real/pseudo-bonsai --split val --output runtime/bonsai-validation.json
```

Test остаётся закрыт до фиксации конфигурации. Не смешивайте результаты разных сигнатур в одной папке и не удаляйте маркеры test для повторного подбора.
