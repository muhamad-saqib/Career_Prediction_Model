# <h1><b><u>Career Prediction</h1>

# from google.colab import drive
# drive.mount('/content/drive')

# path = '/content/drive/MyDrive/ML_Datasets/Career Prediction Dataset 10000+.csv'

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder , StandardScaler , MultiLabelBinarizer
from sklearn.metrics import accuracy_score , classification_report
import pickle
import joblib

# path = '/content/drive/MyDrive/ML_Datasets/Career Prediction Dataset 10000+.csv'

dataset1 = pd.read_csv('career_prediction_dataset_10000_logic.csv')
dataset1.head()
dataset1.tail()
dataset1.shape
dataset1.info()
dataset1.describe()
dataset1.size
dataset1.isnull()

df_copy = dataset1[['Matric Percentage'	,'Intermediate Percentage','Skills', 'Interests','Target Career']]

# Since we dont have any null value in any column of dataset thats why we will not fill any numerical column with median and categorical column with mode while important null columns like skills and interest also wont be fill by blank string' '.
print(df_copy.columns)

scaler = StandardScaler()

df_copy[['Matric Percentage', 'Intermediate Percentage']] = scaler.fit_transform(
    df_copy[['Matric Percentage', 'Intermediate Percentage']]
)

print("Scaled data mean:\n", df_copy[['Matric Percentage', 'Intermediate Percentage']].mean())
print("Scaled data std:\n", df_copy[['Matric Percentage', 'Intermediate Percentage']].std())


# ---------------- Cleaning Function ----------------
def clean_and_convert(cell):
    if pd.isna(cell):  
        return []

    return [x.strip().lower() for x in str(cell).split(",") if x.strip()]

df_copy = dataset1[['Matric Percentage', 'Intermediate Percentage', 'Skills', 'Interests', 'Target Career']]


df_copy['Skills'] = df_copy['Skills'].apply(clean_and_convert)
df_copy['Interests'] = df_copy['Interests'].apply(clean_and_convert)

mlb_skills = MultiLabelBinarizer()
skills_encoded = mlb_skills.fit_transform(df_copy['Skills'])
skills_df = pd.DataFrame(
    skills_encoded,
    columns=[f"Skill_{s}" for s in mlb_skills.classes_]
).reset_index(drop=True)

print("Unique Skills:", mlb_skills.classes_[:10], "...")
print("Encoded Skills Matrix Shape:", skills_encoded.shape)

mlb_interests = MultiLabelBinarizer()
interests_encoded = mlb_interests.fit_transform(df_copy['Interests'])
interests_df = pd.DataFrame(
    interests_encoded,
    columns=[f"Interest_{i}" for i in mlb_interests.classes_]
).reset_index(drop=True)

print("Unique Interests:", mlb_interests.classes_[:10], "...")
print("Encoded Interests Matrix Shape:", interests_encoded.shape)

df_copy = df_copy.reset_index(drop=True)

df_ready = pd.concat(
    [df_copy.drop(['Skills', 'Interests'], axis=1), skills_df, interests_df],
    axis=1
)

print("df_ready shape:", df_ready.shape)
print("df_copy shape:", df_copy.shape)


career_encoder = LabelEncoder()
df_copy['Target Career'] = career_encoder.fit_transform(df_copy['Target Career'])

print("Career Classes Mapping:")
for i, cls in enumerate(career_encoder.classes_):
    print(f"{i} → {cls}")

X = df_ready.drop('Target Career', axis=1)  # Features (Matric, Inter, Skills, Interests)
y = df_copy['Target Career']                # Target only (career encoded)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



joblib.dump(model, "career_model.pkl")
joblib.dump(career_encoder, "career_encoder.pkl")
joblib.dump(mlb_skills, "skills_encoder.pkl")
joblib.dump(mlb_interests, "interests_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")


