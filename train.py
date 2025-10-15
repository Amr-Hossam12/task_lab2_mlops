import os
import joblib
import pandas as pd
from sklearn.svm import SVC

train_df = pd.read_csv('data/train.csv')
X_train = train_df.drop('left', axis=1)
y_train = train_df['left']

model = SVC(kernel='poly', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("SVC model saved to models/model.pkl")
