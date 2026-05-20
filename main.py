from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Bird Classification API")

# Параметры модели
MODEL_PATH  = "best_classification_model.keras"
IMG_SIZE    = 128
CLASS_NAMES = [
    "Asian Green Bee-Eater",
    "Brown-Headed Barbet",
    "Common Kingfisher"
]

# Загрузка модели при старте сервера (один раз)
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Модель загружена: {MODEL_PATH}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Предобработка изображения перед передачей в модель."""
    # Открываем изображение из байтов
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Изменяем размер до 128×128
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Конвертируем в массив и нормализуем [0, 1]
    image_array = np.array(image, dtype="float32") / 255.0

    # Добавляем batch dimension: (128, 128, 3) → (1, 128, 128, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.get("/")
def root():
    """Проверка работоспособности API."""
    return {"status": "ok", "message": "Bird Classification API is running"}


@app.get("/health")
def health():
    """Health check эндпоинт."""
    return {"status": "healthy", "model": MODEL_PATH, "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Классификация изображения птицы.

    Принимает изображение (jpg, png и др.),
    возвращает предсказанный класс и вероятности по всем классам.
    """
    # Читаем байты загруженного файла
    image_bytes = await file.read()

    # Предобработка
    try:
        image_array = preprocess_image(image_bytes)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Не удалось обработать изображение"})

    # Предсказание
    predictions = model.predict(image_array, verbose=0)[0]

    # Формируем ответ
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = float(predictions[predicted_index])

    # Вероятности по всем классам
    probabilities = {
        cls: round(float(prob), 6)
        for cls, prob in zip(CLASS_NAMES, predictions)
    }

    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 6),
        "probabilities":   probabilities
    })