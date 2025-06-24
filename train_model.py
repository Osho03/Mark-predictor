# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 📊 Sample training data (expand this later for better accuracy)
data = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5, 6, 7],
    "sleep": [5, 6, 7, 8, 5, 6, 7],
    "focus": [1, 2, 2, 3, 3, 1, 2],
    "distraction": [4, 3, 2, 1, 5, 6, 2],
    "marks": [40, 50, 60, 80, 70, 35, 65]
})

# 🎯 Features & Target
X = data[["hours", "sleep", "focus", "distraction"]]
y = data["marks"]

# 🧠 Train model
model = LinearRegression()
model.fit(X, y)

# 💾 Save model
pickle.dump(model, open("model.pkl", "wb"))
print("✅ Model saved successfully.")