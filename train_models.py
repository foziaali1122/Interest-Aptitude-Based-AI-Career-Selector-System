import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# -----------------------------
# ML MODEL (Career Prediction)
# -----------------------------

# Dummy dataset
data = {
    "gpa": np.random.uniform(2.0, 4.0, 200),
    "python_skill": np.random.randint(0, 2, 200),
    "design_skill": np.random.randint(0, 2, 200),
    "math_skill": np.random.randint(0, 2, 200),
    "career": np.random.choice(["Data Scientist", "Web Developer", "AI Engineer"], 200)
}

df = pd.DataFrame(data)

X = df.drop("career", axis=1)
y = df["career"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump((model, le), "ml_model.pkl")

# -----------------------------
# DL MODEL (Resume Classification)
# -----------------------------

X_dl = np.random.rand(500, 10)
y_dl = np.random.randint(0, 3, 500)

y_dl_cat = to_categorical(y_dl)

dl_model = Sequential([
    Dense(64, activation="relu", input_shape=(10,)),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

dl_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
dl_model.fit(X_dl, y_dl_cat, epochs=5, verbose=1)

dl_model.save("dl_model.h5")

print("✅ Models Trained & Saved")
