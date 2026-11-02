# Sentiment Analysis с BERT

## Описание
Проект по анализу тональности отзывов о фильмах с использованием трансформеров. Модель — `distilbert-base-uncased`, данные — 2000 отзывов из датасета IMDB (два класса: 0 — негатив, 1 — позитив).

Проект выполнен за 7 дней:

| День | Тема | Файл |
|---|---|---|
| 1 | Токенизация | `day01_tokenization.ipynb` |
| 2 | Hidden states | `day02_tokenization.ipynb` |
| 3 | Визуализация attention | `day03_tokenization.ipynb` |
| 4 | Baseline: CLS-эмбеддинги + LogisticRegression | `day04_05_06_07tokenization.ipynb` |
| 5 | Fine-tuning DistilBERT | `day04_05_06_07tokenization.ipynb` |
| 6 | Сравнение baseline и fine-tuned | `day04_05_06_07tokenization.ipynb` |
| 7 | Анализ ошибок и демо-приложение | `day04_05_06_07tokenization.ipynb`, `app.py` |

## Структура
- `data/imdb_2000.csv` — данные (2000 отзывов IMDB, классы сбалансированы)
- `fine_tuned_model/` — обученная модель (не хранится в git из-за размера, создаётся в Дне 5)
- `baseline_model.pkl` — baseline: LogisticRegression на CLS-эмбеддингах
- `app.py` — Gradio-приложение
- `baseline_results.txt`, `fine_tuned_results.txt` — метрики моделей
- `comparison_results.txt` — сравнение моделей
- `error_analysis.txt` — анализ ошибок
- `confusion_matrix_*.png`, `attention_*.png` — графики

## Запуск демо

Сначала нужна обученная модель в папке `fine_tuned_model/`: выполните ноутбук до конца Дня 5 (около 5 минут на CPU).

```bash
pip install gradio transformers torch
python app.py
```
Откройте http://127.0.0.1:7860

## Результаты

Обе модели оценивались на одних и тех же 400 отзывах (split 80/20, `random_state=42`, `stratify`).

### Fine-tuned модель (DistilBERT, 3 эпохи, lr=2e-5, batch 16, max_length=128):
- F1 (macro): 0.85
- Accuracy: 0.85

### Baseline модель (CLS-эмбеддинги DistilBERT без дообучения + LogisticRegression):
- F1 (macro): 0.82
- Accuracy: 0.82

Улучшение: +3 п.п. F1 (+3.66% относительно baseline)

### Анализ ошибок
- 60 ошибок из 400: 30 false positives и 30 false negatives — модель не смещена в сторону одного класса.
- Ошибочные отзывы в среднем длиннее: 1760 символов против 1252. Модель видит только первые 128 токенов, поэтому итоговая оценка в конце длинного отзыва часто обрезается.
- Типичные трудные случаи: смешанная тональность и ирония («This movie is ridiculous! That's exactly what I like»), отзывы, где автор сначала хвалит, а потом ругает (или наоборот).

## Использование в коде
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
model.eval()

inputs = tokenizer("Your text here", return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()   # 0 — negative, 1 — positive
```

## Требования
- Python 3.10+ (проект собран на 3.13)
- transformers, torch
- datasets, pandas, scikit-learn
- matplotlib, seaborn
- gradio (для демо)
