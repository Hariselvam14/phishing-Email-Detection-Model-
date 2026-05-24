from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

emails = [
    "Congratulations! You won a free iPhone. Click here now",
    "Your bank account is blocked. Login immediately",
    "Meeting at 3 PM tomorrow",
    "Project submission deadline extended",
    "Urgent! Verify your password now",
    "Lunch at college canteen today"
]

labels = ["Phishing", "Phishing", "Safe", "Safe", "Phishing", "Safe"]


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)


X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_email = ["Click this link to claim your free reward"]
new_email_vector = vectorizer.transform(new_email)
prediction = model.predict(new_email_vector)

print("\nPrediction for new email:", prediction[0])
