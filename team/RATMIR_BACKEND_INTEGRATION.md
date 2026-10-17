# Ратмиру: подключение модели к готовому backend

Обновление 17 октября: backend, synthetic demo и native Bonsai уже работают как пользовательские systemd-службы в `/home/angel/forte-review-service`. Сначала прочитай [текущее развёртывание](../docs/linux-deployment.md) и проверь существующие службы. Инструкции ниже описывают начальную подготовку нового хоста. Сетевые проверки TCP/TLS на 7844 теперь прошли на обоих ноутбуках; публичный API пока не подтверждён из-за отказа автоматической проверки запуска туннеля. Качество модели остаётся недостаточным для соревнования.

Этот файл можно целиком передать своему Codex в локальной копии проекта на RTX 4060. Это продолжение ML-задачи; backend уже обновлён. Модель пока не объявлена готовой.

Прочитай AGENTS.md, docs/ml-handoff.md и docs/backend-handoff.md. Сохрани текущие изменения; получи актуальный main без reset/force и без потери незакоммиченной работы. Используй уже рабочую CUDA-среду, если она есть. Не переустанавливай драйвер/torch просто из-за различия с README. Все реальные видео, кадры, наблюдения и ответы модели остаются на локальном компьютере; в чат выводи только агрегаты.

## Сначала проверить публичную сеть, не дожидаясь модели

Владелец решил провести этот этап на твоём ноутбуке: на его сети соединение Cloudflare на 7844 не работает ни через QUIC, ни через HTTP/2. Разрешение организаторов на Cloudflare подтверждено. Пока используй только синтетические видео и отдельные ключ/порт/runtime. Linux, из корня проекта, с существующей core-средой:

```bash
.venv/bin/python -m scripts.make_demo
mkdir -p runtime/public-demo
if [ ! -f runtime/public-demo/access-key.txt ]; then
  (umask 077; .venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(24))' > runtime/public-demo/access-key.txt)
fi
export FORTE_API_KEY="$(cat runtime/public-demo/access-key.txt)"
export FORTE_ENGINE=demo
export FORTE_MODEL=models/demo.npz
export FORTE_RUNTIME=runtime/public-demo
.venv/bin/python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

Во втором терминале проверь `curl -f http://127.0.0.1:8001/ready`, затем запусти установленный из официального источника `cloudflared tunnel --url http://127.0.0.1:8001 --no-autoupdate`. Если QUIC не соединяется, перезапусти с `--protocol http2`. Наличие URL не означает успех: дождись подключения и проверь публичный `/ready`. В третьем терминале из корня проекта:

```bash
export FORTE_API_KEY="$(cat runtime/public-demo/access-key.txt)"
.venv/bin/python -m scripts.smoke_api data/demo/multiple.mov --url https://YOUR-HOST.trycloudflare.com
```

Передай владельцу URL и ключ приватно для повторного POST/GET с другого подключения. Не меняй firewall/VPN вслепую и не публикуй данные настоящих заданий на этом демосервере. Если сеть тоже блокирует 7844, зафиксируй ошибку и сообщи владельцу. Команды Windows находятся в docs/backend-handoff.md.

## Затем подключить настоящую модель

1. Заверши выбранный кандидат. Зафиксируй движок, путь, ревизию/хеши весов, Python и зависимости. Проверяй импорт, CUDA и одну настоящую короткую обработку. Не считать HTTP stub готовой моделью. Если Qwen/Bonsai не заработал, не выдавать слабые бинарные веса за semantic reference.
2. Запусти API из той же среды, где работает модель, с отдельным runtime и новым приватным ключом. Linux-пример для Qwen ниже. Путь `.venv-inference/bin/python` замени только если фактическая рабочая среда другая.

```bash
mkdir -p runtime/jury
if [ ! -f runtime/jury/access-key.txt ]; then
  (umask 077; .venv-inference/bin/python -c 'import secrets; print(secrets.token_urlsafe(24))' > runtime/jury/access-key.txt)
fi
export FORTE_API_KEY="$(cat runtime/jury/access-key.txt)"
export FORTE_ENGINE=qwen
export FORTE_MODEL=models/qwen3-vl-2b
export FORTE_RUNTIME=runtime/jury
export FORTE_READINESS_TIMEOUT=90
.venv-inference/bin/python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Для Windows смотри scripts/serve.ps1. Для проверенного semantic NPZ используй `reference` и его реальный путь. Для Bonsai сначала запусти локальный llama.cpp server с готовыми GGUF и проектором, затем существующий `python -m scripts.serve_bonsai --config models/bonsai/inference.json --runtime runtime/jury --port 8000`. Установи FORTE_API_KEY в том же shell. Bonsai server остаётся отдельным процессом; управляй его завершением отдельно. Не направляй inference на публичный URL.

3. Во втором терминале проверь `curl -f http://127.0.0.1:8000/ready`. Сначала сервер делает реальную загрузку и синтетический inference; 503 означает, что можно исследовать ошибку локально, но нельзя подавать эту конфигурацию жюри. После изменения модели перезапусти API. Не подменять проверку readiness флагом или мок-ответом.
4. Локально запусти `python -m scripts.smoke_api <путь-к-разрешённому-MOV>` с тем же FORTE_API_KEY. Возьми минимум один известный положительный и один отрицательный train-пример, затем много событий и самый длинный ролик. При наличии отдельной среды используй её полный путь к Python. Сохрани только агрегаты: verdict, число событий/кадров, полное время, ошибки. Люди самостоятельно проверяют содержание кадров и применимость правил.
5. Сделай три холодных API-запуска длинного ролика, включая upload → polling → done, измерь максимальную VRAM. Цель <240 с, предел 300 с. Не путать это с ранее измеренным быстрым weak-классификатором. Не менять split и не смотреть отложенный test ради подбора параметров.
6. Проверь локальный inference без внешней сети после полной загрузки зависимостей/весов. Для Bonsai loopback должен работать. Затем верни соединение и запусти разрешённый `cloudflared tunnel --url http://127.0.0.1:8000`. Передай владельцу URL и ключ приватно, а в Git — только конфигурацию без секретов, хеши и агрегированный отчёт. Владелец проверит POST/GET с другого подключения и выполнит подачу.

Ожидаемый результат передачи: движок + локальный путь + хеши, готовый `/ready`, проверенные positive/negative/multiple результаты, три полных времени длинного видео, VRAM, ограничения и точная команда запуска. Если отсутствуют готовые веса или projector/runtime, явно назвать оставшийся файл/ошибку; не объявлять интеграцию завершённой.
