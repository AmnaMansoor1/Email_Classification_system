from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext

from email_classifier import build_trained_classifier


class EmailClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Email Classification System")
        self.root.geometry("700x500")

        training_file = Path("training_emails.json")
        self.classifier = build_trained_classifier(training_file)

        self.title_label = tk.Label(
            root,
            text="Email Classification System",
            font=("Arial", 16, "bold"),
        )
        self.title_label.pack(pady=(15, 10))

        self.instructions_label = tk.Label(
            root,
            text="Enter an email below and click Classify Email.",
            font=("Arial", 11),
        )
        self.instructions_label.pack()

        self.email_input = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            width=80,
            height=14,
            font=("Arial", 11),
        )
        self.email_input.pack(padx=20, pady=15, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=5)

        self.classify_button = tk.Button(
            self.button_frame,
            text="Classify Email",
            width=18,
            command=self.classify_email,
        )
        self.classify_button.pack(side=tk.LEFT, padx=8)

        self.clear_button = tk.Button(
            self.button_frame,
            text="Clear",
            width=12,
            command=self.clear_input,
        )
        self.clear_button.pack(side=tk.LEFT, padx=8)

        self.result_label = tk.Label(
            root,
            text="Predicted Category: ",
            font=("Arial", 12, "bold"),
        )
        self.result_label.pack(pady=(15, 5))

    def classify_email(self) -> None:
        email_text = self.email_input.get("1.0", tk.END).strip()
        if not email_text:
            messagebox.showwarning("Input Required", "Please enter an email before classifying.")
            return

        predicted_category = self.classifier.predict(email_text)
        self.result_label.config(text=f"Predicted Category: {predicted_category}")

    def clear_input(self) -> None:
        self.email_input.delete("1.0", tk.END)
        self.result_label.config(text="Predicted Category: ")


def main() -> None:
    root = tk.Tk()
    app = EmailClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
