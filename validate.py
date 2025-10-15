import json
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Load test data and model 
test_df = pd.read_csv('data/test_encoded.csv')
X_test = test_df.drop('left', axis=1)
y_test = test_df['left']
model = joblib.load('models/model.pkl')

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Save test metrics 
with open('metrics.json', 'w') as f:
    json.dump({'test_accuracy': acc}, f)
print(f"Test Accuracy: {acc:.4f}")
print("Saved metrics.json")

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - HR Analytics')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")
