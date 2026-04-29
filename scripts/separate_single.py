import sys
from pathlib import Path
import demucs.api
import librosa
import torch

def separate_and_save(input_file):
    """
    Функция инференса для разделения аудио на 4 
    стэма с использованием htdemucs.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Файл '{input_file}' не найден.")
        return

    out_dir = Path("separated")
    out_dir.mkdir(exist_ok=True)

    y, sr = librosa.load(input_path, sr=None, mono=False)

    # Обрезаем тишину; ставим 50, чтобы не обрезало затухание в конце песен
    y_trimmed, _ = librosa.effects.trim(y, top_db=50)

    mix_tensor = torch.tensor(y_trimmed, dtype=torch.float32)

    separator = demucs.api.Separator(model="htdemucs")
    _, separated = separator.separate_tensor(mix_tensor, sr=sr)

    for stem_name, source_tensor in separated.items():
        output_name = f"{stem_name}_{input_path.stem}.wav"
        demucs.api.save_audio(source_tensor, str(out_dir / output_name),
                              samplerate=separator.samplerate)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Укажите путь к аудиофайлу")
    else:
        separate_and_save(sys.argv[1])