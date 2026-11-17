# Что делать с CREMA-D

## Результат проверки вашего архива

Проверен `D:\Downloads\CREMA-D-master.zip` без распаковки и изменения исходника:

- размер ZIP: 23 391 827 байт;
- в `AudioWAV` найдено 7 442 файла с расширением `.wav`;
- **все 7 442 — указатели Git LFS**, настоящих WAV-заголовков нет;
- типичный файл занимает 130 байт и начинается с `version https://git-lfs.github.com/spec/v1`;
- сумма размеров настоящих аудиозаписей, указанных в этих pointers: 605 899 936 байт, примерно 606 MB или 578 MiB;
- по именам представлены 91 диктор и 6 классов. Это инвентаризация имён, не аудит аудио.

Подробный машинный отчёт: `reports/crema_zip_audit.json`. Сам ZIP распаковывать для обучения сейчас бессмысленно: реальные сигналы в нём отсутствуют. Такой эффект загрузки ZIP описан в [README CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D).

## 1. Получить настоящие AudioWAV через Git LFS

На этом компьютере Git LFS уже установлен, проверена версия 3.7.1. Текущая копия `AudioWAV` загружена и прошла аудит. Для загрузки в новой рабочей копии выполнить команды ниже из папки проекта в PowerShell. Папка `data/raw/CREMA-D` должна ещё не существовать.

```powershell
New-Item -ItemType Directory -Path data/raw -Force | Out-Null

# При клонировании пока не скачиваем большие аудио и видео.
$previousLfsSmudge = $env:GIT_LFS_SKIP_SMUDGE
try {
    $env:GIT_LFS_SKIP_SMUDGE = '1'
    git clone --depth 1 https://github.com/CheyneyComputerScience/CREMA-D.git data/raw/CREMA-D
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось клонировать CREMA-D.' }
}
finally {
    if ($null -eq $previousLfsSmudge) {
        Remove-Item Env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
    }
    else {
        $env:GIT_LFS_SKIP_SMUDGE = $previousLfsSmudge
    }
}

# Настраиваем LFS только в этом клоне и скачиваем только WAV.
git -C data/raw/CREMA-D lfs install --local
git -C data/raw/CREMA-D lfs pull --include='AudioWAV/*' --exclude=
```

Последний аргумент `--exclude=` очищает исключения для этой команды. Фильтры `--include`/`--exclude` документированы в [git-lfs-pull](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-pull.adoc). MP3 и видео останутся указателями; для нашей задачи это нормально. Скачивание требует сети и свободного места: WAV хранятся и в рабочей копии, и в локальном LFS-кеше, поэтому одного объёма 606 MB на диске недостаточно; выделить хотя бы около 2 GB под клон и служебные данные, отдельно место под будущие кеши и модели.

Если клон уже есть или загрузка прервалась, не удалять его и не клонировать поверх. Проверить адрес:

```powershell
git -C data/raw/CREMA-D remote get-url origin
git -C data/raw/CREMA-D lfs pull --include='AudioWAV/*' --exclude=
```

Продолжать только если `origin` соответствует выбранному CREMA-D. Если GitHub не выдаёт LFS-объекты, использовать официальное зеркало, указанное в README датасета, или выяснить ошибку сети/квоты. Переименование pointers или их ручная распаковка не создаёт аудио.

## 2. Проверить, что скачались аудиоданные

```powershell
.\.venv\Scripts\python.exe tools/check_crema.py data/raw/CREMA-D/AudioWAV --report reports/crema_audio_headers.json
```

Ожидается `wav_header: 7442`, отсутствие `lfs_pointer` и `complete_wav_headers: true`. Проверка заголовков не заменяет декодирование. Код возврата 2 означает, что набор не прошёл предварительную проверку; это ожидаемый результат для текущего ZIP.

Повторить проверку исходного архива можно так:

```powershell
.\.venv\Scripts\python.exe tools/check_crema.py 'D:\Downloads\CREMA-D-master.zip' --report reports/crema_zip_audit.json
```

После получения WAV открыть несколько файлов в GUI через `run_gui.bat`, проверить воспроизведение и спектрограммы. GUI теперь отдельно объясняет ошибку, если вместо WAV выбран LFS-pointer.

## 3. Аудит реального аудио и metadata.csv

Скрипт декодирует каждый WAV через SoundFile. Он извлекает метку и диктора из имени, измеряет длительность и RMS после сведения каналов в моно, вычисляет SHA-256 файла и записывает решение по каждой записи.

```powershell
$cremaRevision = git -C data/raw/CREMA-D rev-parse HEAD
.\.venv\Scripts\python.exe -m tools.audit_crema data/raw/CREMA-D/AudioWAV --revision $cremaRevision
```

Из 7 442 проверенных файлов пригодны 7 435 записей от 91 диктора. У `1076_MTI_SAD_XX.wav` RMS равен нулю. Ещё шесть записей образуют три пары побайтно одинаковых файлов с разными эмоциями. Обе записи каждой пары помечены как исключённые. SHA-256 всех скачанных файлов совпал с контрольными суммами Git LFS.

Порог исключения по длительности составляет 0,25 с, порог RMS — `0.0001` в нормализованных отсчётах. Записи дольше 8 секунд получают отметку для просмотра. Для побайтных копий с согласованными метками скрипт сохраняет первую пригодную запись по порядку имени.

- `data/metadata.csv` содержит все записи и поле `valid`. Значения `relative_path` отсчитываются от папки `AudioWAV`.
- `data/exclusions.csv` содержит исключённые записи и причины.
- `data/labels.json` фиксирует порядок шести классов.
- `data/crema_audio_audit.json` содержит версию CREMA-D, пороги, группы дубликатов и итоговые количества.

Для разбиения взять из `metadata.csv` строки с `valid=True`.

## 4. Зафиксировать разбиения

Реализовано в первом этапе полного проекта. Speaker-split выделяет 63/14/14 дикторов для train/validation/test.
Random-split использует те же количества файлов и стратификацию по эмоциям. Все окна и шумовые версии
одной записи остаются в той же части. Статистики нормализации вычисляются только по train.

```powershell
.\.venv\Scripts\python.exe -m tools.fetch_background
.\.venv\Scripts\python.exe -m tools.prepare_data
```

Команда также рассчитывает Log-Mel и MFCC всех пригодных записей, признаки для SVM и шумовые тестовые
рецепты. Выходной каталог указан в `data/prepared/current.json`, результаты — в `reports/stage1.md`.
Подробное описание файлов и программного доступа находится в `README.md`.

## 5. Обучение трёх моделей и эксперименты

1. Использовать готовые признаки и train-статистики через `ser.dataset.PreparedDataset`.
2. Реализовать CNN на PyTorch, GMM с собственным EM на NumPy и SVM на scikit-learn.
3. Отладить модели и выбрать гиперпараметры по validation. Test не используется для выбора.
4. Обучить финальные модели для обоих протоколов, сохранить параметры и checkpoints.
5. Провести общую чистую и шумовую оценку, сохранить predictions, метрики и графики.
6. Подключить модели к GUI и включить распознавание.

Подробный план из трёх этапов, параметры, формулы и бюджет запусков — в `PLAN.md`. GUI остаётся на Qt Widgets.
