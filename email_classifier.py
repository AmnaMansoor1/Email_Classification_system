import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


class EmailClassifier:
    def __init__(self) -> None:
        self.stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "have",
            "i",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "our",
            "the",
            "this",
            "to",
            "was",
            "we",
            "with",
            "you",
            "your",
        }
        self.categories: list[str] = []
        self.vocabulary: set[str] = set()
        self.class_document_counts: Counter = Counter()
        self.class_word_counts: dict[str, Counter] = defaultdict(Counter)
        self.class_total_words: Counter = Counter()
        self.total_documents = 0
        self.is_trained = False

    def preprocess(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return [token for token in text.split() if token and token not in self.stop_words]

    def train(self, training_data: list[dict]) -> None:
        self.categories = sorted({item["category"] for item in training_data})
        self.total_documents = len(training_data)
        self.class_document_counts.clear()
        self.class_word_counts = defaultdict(Counter)
        self.class_total_words.clear()
        self.vocabulary.clear()

        for item in training_data:
            category = item["category"]
            tokens = self.preprocess(item["email"])
            self.class_document_counts[category] += 1
            self.class_word_counts[category].update(tokens)
            self.class_total_words[category] += len(tokens)
            self.vocabulary.update(tokens)

        self.is_trained = True

    def predict(self, email_text: str) -> str:
        if not self.is_trained:
            raise ValueError("The classifier must be trained before prediction.")

        tokens = self.preprocess(email_text)
        if not tokens:
            return "Inquiry"

        vocabulary_size = max(len(self.vocabulary), 1)
        category_scores: dict[str, float] = {}

        for category in self.categories:
            prior = self.class_document_counts[category] / self.total_documents
            score = math.log(prior)
            total_words = self.class_total_words[category]

            for token in tokens:
                token_count = self.class_word_counts[category][token]
                likelihood = (token_count + 1) / (total_words + vocabulary_size)
                score += math.log(likelihood)

            category_scores[category] = score

        return max(category_scores, key=category_scores.get)


def load_json(file_path: Path) -> list[dict]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def build_trained_classifier(training_file: Path) -> EmailClassifier:
    classifier = EmailClassifier()
    training_data = load_json(training_file)
    classifier.train(training_data)
    return classifier


def classify_emails_from_file(input_file: Path, training_file: Path) -> list[dict]:
    classifier = build_trained_classifier(training_file)
    emails = load_json(input_file)

    results = []
    for item in emails:
        results.append(
            {
                "id": item["id"],
                "email": item["email"],
                "predicted_category": classifier.predict(item["email"]),
            }
        )

    return results


if __name__ == "__main__":
    input_file = Path("sample_emails.json")
    training_file = Path("training_emails.json")
    classified_emails = classify_emails_from_file(input_file, training_file)

    print("Email Classification Results")
    print("-" * 30)
    for item in classified_emails:
        print(f'Email {item["id"]}: {item["predicted_category"]}')
        print(f'Text: {item["email"]}')
        print()
