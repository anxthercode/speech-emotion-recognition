"""Generate figures and a Russian project report from the held-out evaluation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ser.dataset import PreparedDataset, read_csv
from ser.splits import LABELS

NAMES = {"svm": "SVM", "gmm": "GMM", "cnn": "CNN"}
SPLITS = {"speaker": "По дикторам", "random": "Случайное"}
COLORS = {"svm": "#0072B2", "gmm": "#D55E00", "cnn": "#009E73"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(fig, folder, name):
    fig.savefig(folder / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plots(folder, evaluation):
    rows = evaluation["summary"]
    groups = {(r["model"], r["split"], r["condition"], r["snr_db"]): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, layout="constrained")
    for row, split in enumerate(SPLITS):
        for column, (kind, title) in enumerate((("white", "Белый шум"), ("background", "Шум кондиционера"))):
            ax = axes[row, column]
            for model, name in NAMES.items():
                conditions = [groups[model, split, kind, snr] for snr in (0, 10, 20)] + [groups[model, split, "clean", None]]
                ax.errorbar(range(4), [r["macro_f1"] for r in conditions], marker="o", capsize=3,
                            yerr=[r["macro_f1_std"] or 0 for r in conditions], color=COLORS[model], label=name)
            ax.set(title=f"{SPLITS[split]} · {title}", ylabel="Macro-F1", ylim=(0, .7),
                   xticks=range(4), xticklabels=["0 дБ", "10 дБ", "20 дБ", "Чистая"])
            ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(loc="upper left", ncol=3, fontsize=8)
    save(fig, folder, "noise_robustness.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for row, split in enumerate(SPLITS):
        for column, (model, name) in enumerate(NAMES.items()):
            ax = axes[row, column]
            matrix = np.mean([r["confusion_normalized"] for r in evaluation["results"]
                              if (r["model"], r["split"], r["condition"]) == (model, split, "clean")], axis=0)
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            ax.set(title=f"{name} · {SPLITS[split]}", xlabel="Предсказание", ylabel="Истинный класс",
                   xticks=range(6), yticks=range(6), xticklabels=list(LABELS), yticklabels=list(LABELS))
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", color="white" if value > .55 else "black", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=.75, label="Доля внутри истинного класса")
    save(fig, folder, "confusion_clean.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for run in evaluation["runs"]:
        if run["model"] == "cnn":
            history = read_json(Path(run["run_directory"]) / "history.json")
            ax = axes[list(SPLITS).index(run["split"])]
            ax.plot([r["epoch"] for r in history], [r["validation_macro_f1"] for r in history], label=f"seed {run['seed']}")
    for ax, split in zip(axes, SPLITS):
        ax.set(title=SPLITS[split], xlabel="Эпоха", ylabel="Validation Macro-F1")
        ax.legend()
    save(fig, folder, "cnn_learning.png")

    run = next(r for r in evaluation["runs"] if r["model"] == "gmm" and r["split"] == "speaker")
    histories = read_json(Path(run["run_directory"]) / "em_history.json")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for group in histories:
        history = group["initializations"][group["selected_initialization"]]["history"]
        ax.plot([r["iteration"] for r in history], [r["mean_log_likelihood"] for r in history], label=list(LABELS)[group["class"]])
    ax.set(xlabel="Итерация EM", ylabel="Средний log-likelihood на кадр", title=f"GMM · по дикторам · seed {run['seed']}")
    ax.legend(ncol=3)
    save(fig, folder, "gmm_learning.png")

    data = PreparedDataset(split="speaker", subset="test")
    file_id = data.records[0]["file_id"]
    matrices = [data.features(file_id)["log_mel"][0], data.features(file_id, "white", 0)["log_mel"][0]]
    maximum = max(float(a.max()) for a in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, matrix, title in zip(axes, matrices, ("Чистая речь", "Белый шум · 0 дБ")):
        image = ax.imshow(matrix, origin="lower", aspect="auto", extent=(0, 4, 0, 64), vmin=maximum - 80, vmax=maximum)
        ax.set(title=title, xlabel="Время, с", ylabel="Mel-полоса")
    fig.colorbar(image, ax=axes, label="Мощность, дБ")
    fig.suptitle(file_id)
    save(fig, folder, "spectrogram_noise.png")

    metadata = [r for r in read_csv("data/metadata.csv") if r["valid"] == "True"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].bar(list(LABELS), [sum(r["emotion"] == code for r in metadata) for code in LABELS])
    axes[0].set(ylabel="Число записей", title="Эмоции в подготовленном корпусе")
    axes[1].hist([float(r["duration_s"]) for r in metadata], bins=30)
    axes[1].set(xlabel="Длительность, с", ylabel="Число записей", title="Длительность записей")
    save(fig, folder, "dataset_distribution.png")


def report(folder, evaluation):
    rows = evaluation["summary"]
    clean = [r for r in rows if r["condition"] == "clean"]
    lines = ["# Распознавание эмоций по голосу на CREMA-D", "",
             "Галкин В. А. и Леончук Е. Д. · Основы машинного обучения", "",
             "## Задача и данные", "",
             "По записи речи требуется определить одну из шести эмоций. После аудита осталось 7 435 WAV от 91 диктора. "
             "В каждом протоколе 5 151 запись для обучения, 1 142 для валидации и 1 142 для теста. "
             "Разбиение по дикторам разделяет голоса между частями. Случайное разбиение стратифицировано по эмоциям.", "",
             "## Признаки и модели", "",
             "Сигнал переводится в моно и 16 кГц. После оконного преобразования Фурье спектральная мощность "
             "суммируется треугольными Mel-фильтрами, затем переводится в дБ. CNN получает окна 64 × 397, "
             "обучается с кросс-энтропией и усредняет softmax по окнам записи. Дискретное косинусное преобразование "
             "даёт MFCC. С производными первого и второго порядка получается 39 признаков на кадр.", "",
             "GMM обучает отдельную смесь диагональных гауссиан на класс. E-шаг вычисляет ответственности компонент, "
             "M-шаг обновляет веса, средние и дисперсии. Предсказание максимизирует логарифмическую оценку записи "
             "при равных априорных вероятностях. SVM с ядром RBF использует среднее и стандартное отклонение "
             "каждого признака по кадрам. Стандартизация опирается на обучающую выборку.", "",
             "Для диагональной смеси плотность кадра равна `p(x|c) = Σₖ πₖ N(x|μₖ, diag(σ²ₖ))`. "
             "Ответственность на E-шаге `rᵢₖ = πₖ N(xᵢ|μₖ, diag(σ²ₖ)) / p(xᵢ|c)`. "
             "На M-шаге вычисляются `Nₖ = Σᵢ rᵢₖ`, `πₖ = Nₖ/N` и `μₖ = Σᵢ rᵢₖxᵢ/Nₖ`. Дисперсия считается с теми же весами вокруг нового среднего. "
             "Нижняя граница дисперсии и расчёт в логарифмах стабилизируют алгоритм. "
             "RBF-ядро SVM задаётся формулой `exp(-γ‖x−z‖²)`.", "",
             "## Эксперимент", "",
             "Гиперпараметры выбраны на валидации разбиения по дикторам. Для каждого разбиения оценены три запуска "
             "CNN и GMM и один запуск SVM. Шум добавлен к одним и тем же тестовым записям при SNR 20, 10 и 0 дБ. "
             "Коэффициент шума вычисляется как `sqrt(P_signal / (P_noise * 10**(SNR/10)))`. "
             "Macro-F1 усредняет F1 шести классов. UAR усредняет полноту шести классов. "
             "Коды ANG, DIS, FEA, HAP, NEU, SAD означают гнев, отвращение, страх, радость, нейтральное состояние и грусть.", "",
             "Выбранные гиперпараметры финальных моделей", ""]
    for run in evaluation["runs"]:
        if run["split"] != "speaker" or run["seed"] != evaluation["runs"][0]["seed"]:
            continue
        p = run["parameters"]
        keys = {"svm": ("C", "gamma"), "gmm": ("n_components", "variance_floor"),
                "cnn": ("learning_rate", "dropout", "weight_decay", "batch_size")}[run["model"]]
        lines.append(f"- {NAMES[run['model']]} · " + ", ".join(f"{key} = {p[key]}" for key in keys))
    lines += ["",
             "## Чистая тестовая выборка", "",
             "| Модель | Разбиение | Macro-F1 | UAR | Accuracy |", "|---|---|---:|---:|---:|"]
    for r in clean:
        values = [f"{r[k]:.3f}" + (f" ± {r[k + '_std']:.3f}" if r[k + "_std"] is not None else "")
                  for k in ("macro_f1", "uar", "accuracy")]
        lines.append(f"| {NAMES[r['model']]} | {SPLITS[r['split']]} | " + " | ".join(values) + " |")
    lines += ["", "Разброс указан как выборочное стандартное отклонение между запусками.", "",
              "## Устойчивость к шуму", "",
              "| Модель | Разбиение | F1, белый шум 0 дБ | Падение, п.п. | F1, фон 0 дБ | Падение, п.п. |",
              "|---|---|---:|---:|---:|---:|"]
    for base in clean:
        white, background = [next(r for r in rows if r["model"] == base["model"] and r["split"] == base["split"]
                                  and r["condition"] == kind and r["snr_db"] == 0) for kind in ("white", "background")]
        lines.append(f"| {NAMES[base['model']]} | {SPLITS[base['split']]} | {white['macro_f1']:.3f} | "
                     f"{white['f1_drop_pp']:.1f} | {background['macro_f1']:.3f} | {background['f1_drop_pp']:.1f} |")
    lines += ["", "## Выводы и ограничения", ""]
    for split, title in SPLITS.items():
        best = max((r for r in clean if r["split"] == split), key=lambda r: r["macro_f1"])
        lines.append(f"- {title}. На чистой речи максимальный Macro-F1 у {NAMES[best['model']]} ({best['macro_f1']:.3f}).")
    for kind, name in (("white", "При белом шуме"), ("background", "При шуме кондиционера")):
        best = max((r for r in rows if r["split"] == "speaker" and r["condition"] == kind and r["snr_db"] == 0),
                   key=lambda r: r["macro_f1"])
        lines.append(f"- {name} 0 дБ на новых дикторах лидирует {NAMES[best['model']]} ({best['macro_f1']:.3f}).")
    lines += ["- Состав теста различается между разбиениями. Разность результатов включает влияние состава записей.",
              "- Результаты характеризуют постановочную англоязычную речь. Для шума использован один фон кондиционера "
              "и одна реализация каждой смеси.", "", "## Время обработки", ""]
    with (folder / "timings.csv").open(encoding="utf-8-sig", newline="") as stream:
        timings = list(csv.DictReader(stream))
    keys = ("model", "split", "seed", "condition", "snr_db")
    counts = {tuple(str(r[k]) if r[k] is not None else "" for k in keys): r["n_test"] for r in evaluation["results"]}
    lines += ["| Модель | Устройство | Инференс, мс на запись |", "|---|---|---:|"]
    for model, name in NAMES.items():
        selected = [r for r in timings if r["model"] == model]
        milliseconds = np.mean([float(r["inference_seconds"]) * 1000 /
                                counts[tuple(r[k] for k in keys)] for r in selected])
        lines.append(f"| {name} | {selected[0]['device'].upper()} | {milliseconds:.2f} |")
    lines += ["", "Замеры выполнены пакетами по 32 записи. Чтение признаков и построение шумовых смесей "
              "замерены отдельно и выполняются один раз на пакет для всех моделей. Таблица сравнивает фактические "
              "режимы запуска с указанными устройствами.", "", "## Файлы результатов", "",
              "[Сводка по всем 42 условиям](test_summary.csv) · [По запускам](test_per_seed.csv) · "
              "[По классам](test_per_class.csv) · [Замеры времени](timings.csv) · [Предсказания](test_predictions.csv)", "",
              "![Устойчивость к шуму](noise_robustness.png)", "", "![Матрицы ошибок](confusion_clean.png)", "",
              "[Распределения данных](dataset_distribution.png) · [Чистая и зашумлённая спектрограммы](spectrogram_noise.png) · "
              "[Кривые CNN](cnn_learning.png) · [Кривые EM](gmm_learning.png)", ""]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("reports/stage3"))
    args = parser.parse_args()
    evaluation = read_json(args.folder / "evaluation.json")
    plots(args.folder, evaluation)
    report(args.folder, evaluation)


if __name__ == "__main__":
    main()
