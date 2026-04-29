import shutil
import zipfile
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import demucs.api
import librosa
import torch

app = FastAPI(title="Music Source Separation API")
separator = demucs.api.Separator(model="htdemucs")

UPLOAD_DIR = Path("temp_storage")
UPLOAD_DIR.mkdir(exist_ok=True)

def check_duration(path, max_sec=300):
    """
    Проверяем длительность аудиофайла.
    """
    duration = librosa.get_duration(path=str(path))
    if duration > max_sec:
        raise HTTPException(
            status_code=400, 
            detail=f"Файл слишком длинный ({duration:.1f} секунд). Доступно: {max_sec} сек."
        )

def validate_file(file: UploadFile):
    """
    Валидация расширения файла.
    """
    allowed_extensions = {".mp3", ".wav", ".flac"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат файла: {file_ext}")

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Принимает аудиофайл, разделяет его на 4 стэма и возвращает ZIP-архив.
    Использует временную директорию. 
    """
    validate_file(file)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / file.filename

        # Сохранение загруженного файла на диск
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        check_duration(input_path)
        
        try:
            # Обрезаем тишину
            y, sr = librosa.load(input_path, sr=None, mono=False)
            y_trimmed, _ = librosa.effects.trim(y, top_db=50)
        
            mix_tensor = torch.tensor(y_trimmed, dtype=torch.float32)
        
            # Инференс на обрезанном тензоре
            _, separated = separator.separate_tensor(mix_tensor, sr=sr)
            
            stem_paths = []
            for stem_name, source_tensor in separated.items():
                stem_file = tmp_dir_path / f"{stem_name}.wav"
                demucs.api.save_audio(source_tensor, str(stem_file), samplerate=separator.samplerate)
                stem_paths.append(stem_file)
            
            # Архивирование результатов
            zip_filename = f"separated_{Path(file.filename).stem}.zip"
            zip_path = UPLOAD_DIR / zip_filename
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for fp in stem_paths:
                    zipf.write(fp, arcname=fp.name)
            
            # Возврат файла
            return FileResponse(
                path=zip_path, 
                filename=zip_filename, 
                media_type='application/zip'
            )
            
        except Exception as e:
            # Логирование ошибки и возврат 500 статуса при сбое модели
            raise HTTPException(status_code=500, detail=f"Ошибка разделения: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)