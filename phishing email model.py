
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training dataset
data = {
    "email": [
        "Your account is suspended. Click here to verify your password",
        "Congratulations! You won a prize. Claim it now",
        "Urgent: Verify your bank account immediately",
        "Click this link to reset your password and avoid suspension",
        "You have won a lottery. Send your bank details",
        "Your package is waiting. Pay the delivery fee now",
        "We detected suspicious login. Verify your account now",
        "Update your payment information immediately",
        "Your account will be blocked. Click to confirm details",
        "You are selected for a cash reward. Claim now",
        "Meeting scheduled for tomorrow at 10 AM",
        "Your assignment has been uploaded successfully",
        "Please find the project report attached",
        "The team meeting starts at 3 PM today",
        "Thank you for submitting your application",
        "Your college timetable has been updated",
        "Your order has been shipped successfully",
        "Please review the document before the meeting",
        "Your exam results are available on the college portal",
        "The monthly newsletter is ready to read",
        "Your password has been changed successfully",
        "Reminder: Your appointment is tomorrow",
        "The invoice for your recent purchase is attached",
        "Please confirm your attendance for the event",
    ],
    "label": [
        "phishing", "phishing", "phishing", "phishing",
        "phishing", "phishing", "phishing", "phishing",
        "phishing", "phishing", "legitimate", "legitimate",
        "legitimate", "legitimate", "legitimate", "legitimate",
        "legitimate", "legitimate", "legitimate", "legitimate",
        "legitimate", "legitimate", "legitimate", "legitimate",
    ]
}

df = pd.DataFrame(data)

# Create and train the machine learning model
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(df["email"], df["label"])

print("=== Phishing Email Detection Model ===")

while True:
    email_text = input(
        "\nEnter email text (or type exit): "
    ).strip()

    if email_text.lower() == "exit":
        print("Program closed.")
        break

    if not email_text:
        print("Please enter some email text.")
        continue

    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    confidence = max(probabilities) * 100

    print("\nPrediction:", prediction.upper())
    print(f"Model confidence: {confidence:.2f}%")

    if prediction == "phishing":
        print("Warning: This email may be a phishing attempt!")
    else:
        print("No phishing pattern strongly detected.")
        print("Still verify the sender and any links.")
