from pathlib import Path

from flask import Flask, render_template, request

from email_classifier import build_trained_classifier


app = Flask(__name__)
training_file = Path("training_emails.json")
classifier = build_trained_classifier(training_file)


@app.route("/", methods=["GET", "POST"])
def index():
    email_text = ""
    predicted_category = None
    error_message = None

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        if email_text:
            predicted_category = classifier.predict(email_text)
        else:
            error_message = "Please enter an email before classifying."

    return render_template(
        "index.html",
        email_text=email_text,
        predicted_category=predicted_category,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=False)
