from transformers import pipeline

print("Loading GoEmotions Transformer... (only the first time)")

classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

print("Model loaded successfully.\n")


def predict_emotions(text, top_k=3):

    predictions = classifier(text)[0]

    predictions = sorted(
        predictions,
        key=lambda x: x["score"],
        reverse=True
    )

    results = []

    for pred in predictions[:top_k]:
        results.append({
            "emotion": pred["label"],
            "confidence": round(float(pred["score"]), 4)
        })

    return results


if __name__ == "__main__":

    while True:

        text = input("\nEnter text (or 'q' to quit): ")

        if text.lower() == "q":
            break

        predictions = predict_emotions(text)

        print("\nTop Predicted Emotions:\n")

        for pred in predictions:
            print(
                f"{pred['emotion']} -> {pred['confidence']}"
            )