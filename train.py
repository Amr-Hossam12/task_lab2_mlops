import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load training data
train_df = pd.read_csv('data/train_encoded.csv')
X_train = train_df.drop('left', axis=1)
y_train = train_df['left']

# Train model
model = DecisionTreeClassifier(max_depth=None, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("✅ Decision Tree model saved to models/model.pkl")
