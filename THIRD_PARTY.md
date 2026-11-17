# Данные и внешние источники

## CREMA-D

Источник: [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D).
Использован commit `1658cd342dff90010aa843eaeebd53610a08b1dc`.
Авторы: Cao H., Cooper D. G., Keutmann M. K., Gur R. C., Nenkova A., Verma R.
Статья: *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*, IEEE Transactions on
Affective Computing, 2014, 5(4), 377–390, DOI `10.1109/TAFFC.2014.2336244`.

README источника указывает ODbL для базы и Database Contents License для её отдельных содержимых:
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/),
[DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
Список исключений проекта не изменяет исходный датасет.

## Фоновый шум MS-SNSD

Используется только файл `noise_test/AirConditioner_1.wav`, 23.448625 секунды, mono, 16 kHz,
из [Microsoft MS-SNSD](https://github.com/microsoft/MS-SNSD).
Commit: `fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742`.
SHA-256: `3fe55d7035d7ef2599841738f818dc4463f5ef983eb4b5fefeee153d8caa2110`.
URL загрузки и машиночитаемая атрибуция находятся в `configs/background_source.json`.

Авторы набора: Reddy C. K. A. et al., *A Scalable Noisy Speech Dataset and Online Subjective Test
Framework*, Interspeech 2019, [статья](https://www.isca-archive.org/interspeech_2019/reddy19_interspeech.pdf).

[README закреплённой версии](https://github.com/microsoft/MS-SNSD/blob/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742/README.md)
указывает два источника шумов: Freesound с CC0 и DEMAND с CC BY-SA 3.0. Соответствия каждого отдельного
WAV его первоначальному источнику в README нет. Поэтому проект сохраняет эту оговорку и обе ссылки,
а не приписывает данному файлу неподтверждённую индивидуальную лицензию:
[CC0](https://creativecommons.org/publicdomain/zero/1.0/),
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
MIT в репозитории относится к коду и не заменяет лицензии аудио.

Источник предназначен только для тестового шума. В Git проекта WAV и производные смеси не включаются.
Для новой установки команда `python -m tools.fetch_background` скачивает файл из закреплённой версии
и сверяет SHA-256. Белый шум генерируется локально и не использует сторонние аудиозаписи.
