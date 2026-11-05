import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model_ft.eval()

# Модель бинарная: 0 — негатив, 1 — позитив (метки IMDB)
label_map = {0: "Negative", 1: "Positive"}


def predict_sentiment(text):
    if not text or not text.strip():
        return "Введите текст для анализа"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())

    result = f"Prediction: {label_map[pred]}\n\nProbabilities:\n"
    for i, prob in enumerate(probs):
        result += f"{label_map[i]}: {prob.item() * 100:.2f}%\n"
    return result


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a movie review in English..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis с BERT",
    description="Fine-tuned DistilBERT на отзывах IMDB: определяет, позитивный отзыв или негативный",
    examples=[
        ["This movie was absolutely fantastic!"],
        ["Terrible, waste of my time."],
        ["It was okay, nothing special."],
    ],
)

if __name__ == "__main__":
    demo.launch()
