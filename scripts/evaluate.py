import numpy as np
import musdb
import demucs.api
import torch
import matplotlib.pyplot as plt
import museval
import time
import sys
import os
from pathlib import Path

sys.path.append("..")

FFMPEG_BIN = str(Path("../tools/ffmpeg/bin").resolve())
os.environ["PATH"] += os.pathsep + FFMPEG_BIN
MUSDB_ROOT = "../musdb18"
N_TRACKS = 5
STEMS = ["vocals", "drums", "bass", "other"]

def evaluate():
    """
    Загружаем тестовый набор MUSDB18, выполняем разделение с помощью Demucs 
    и вычисляем метрики SDR/SIR/SAR для каждого трека. Измеряем скорость инференса. 
    """
    mus = musdb.DB(root=MUSDB_ROOT, subsets="test")
    print(f"Найдено: {len(mus.tracks)} треков, оцениваем: {N_TRACKS}\n")

    separator = demucs.api.Separator(model="htdemucs")
    all_scores = {stem: {"SDR": [], "SIR": [], "SAR": []} for stem in STEMS}

    for track in mus.tracks[:N_TRACKS]:
        print(track.name)

        start_time = time.time()

        #Конвертация и инференс без вычисления градиентов
        mix_tensor = torch.tensor(track.audio.T, dtype=torch.float32)
        with torch.no_grad():
            _, separated = separator.separate_tensor(mix_tensor, sr=track.rate)

        elapsed_time = time.time() - start_time
        print(f"Время обработки: {elapsed_time:.2f} сек.")

        estimates = {s: separated[s].numpy().T for s in STEMS}

        scores = museval.eval_mus_track(track, estimates)
        
        for stem in STEMS:
            # Используем медиану для устойчивости к выбросам или тишине
            sdr_val = np.nanmedian(scores.df.query(f"target == '{stem}' and metric == 'SDR'")["score"])
            sir_val = np.nanmedian(scores.df.query(f"target == '{stem}' and metric == 'SIR'")["score"])
            sar_val = np.nanmedian(scores.df.query(f"target == '{stem}' and metric == 'SAR'")["score"])

            all_scores[stem]["SDR"].append(sdr_val)
            all_scores[stem]["SIR"].append(sir_val)
            all_scores[stem]["SAR"].append(sar_val)

            print(f"   {stem:<8} SDR={sdr_val:+.2f} SIR={sir_val:+.2f} SAR={sar_val:+.2f}")
        print()

    print_summary_table(all_scores)
    plot_results(all_scores)

def print_summary_table(all_scores):
    """
    Выводим в консоль итоговую таблицу со средними 
    значениями по всем протестированным трекам.
    """
    print(f"{'Stem':<10} {'SDR':>8} {'SIR':>8} {'SAR':>8}   (дБ)")
    for stem in STEMS:
        sdr = np.mean(all_scores[stem]["SDR"])
        sir = np.mean(all_scores[stem]["SIR"])
        sar = np.mean(all_scores[stem]["SAR"])
        print(f"{stem:<10} {sdr:>8.2f} {sir:>8.2f} {sar:>8.2f}")

def plot_results(all_scores):
    """
    Визуализируем результаты оценки и сохраням в metrics.png.
    """
    plt.rcParams.update({'font.family': 'arial', 'font.size': 10})
    metrics = ["SDR", "SIR", "SAR"]
    colors = ['#052D6E', '#01939A', '#F4D03F']
    x = np.arange(len(STEMS))
    width = 0.25
    _, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        vals = [np.mean(all_scores[s][metric]) for s in STEMS]
        ax.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(STEMS)
    ax.set_ylabel("дБ")
    ax.set_title(f"Средние метрики по {N_TRACKS} трекам (htdemucs)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("metrics.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    evaluate()