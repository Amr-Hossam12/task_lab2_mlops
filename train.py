import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_df = pd.read_csv('data/train.csv')
X_train = train_df.drop('left', axis=1)
y_train = train_df['left']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("Random Forest model saved to models/model.pkl")
