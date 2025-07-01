#https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset

# %% load dataset and basic data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

# --- Dosyayı oku ---
df = pd.read_csv("heart_attack_prediction_datasett.csv", sep=';', on_bad_lines='skip')


describe = df.describe()

print(df.info())



# %% missing value problem
print(df.isnull().sum())

"""
Patient ID                         0
Age                                0
Sex                                0
Cholesterol                        0
Blood Pressure                     0
Heart Rate                         0
Diabetes                           0
Family History                     0
Smoking                            0
Obesity                            0
Alcohol Consumption                0
Exercise Hours Per Week            0
Diet                               0
Previous Heart Problems            0
Medication Use                     0
Stress Level                       0
Sedentary Hours Per Day            0
Income                             0
BMI                                0
Triglycerides                      0
Physical Activity Days Per Week    0
Sleep Hours Per Day                0
Country                            0
Continent                          0
Hemisphere                         0
Heart Attack Risk                  0
dtype: int64
"""

# %% categorical and numerical feature analysis

"""
Index(['Patient ID', 'Age', 'Sex', 'Cholesterol', 'Blood Pressure',
       'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
       'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
       'Previous Heart Problems', 'Medication Use', 'Stress Level',
       'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
       'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Country',
       'Continent', 'Hemisphere', 'Heart Attack Risk'],
      dtype='object')
"""

categorical_list = ['Sex','Cholesterol','Blood Pressure','Smoking','Heart Attack Risk']

df_categorical = df.loc[:, categorical_list]

save_dir = "/Users/pelinkonak/Desktop/adsız klasör/MakineOgrenmesiProjeleri/grafikler"
os.makedirs(save_dir, exist_ok=True)

for i in categorical_list:
    plt.figure()
    sns.countplot(x=i, data= df_categorical, hue = "Heart Attack Risk")
    plt.title(i)

    plt.savefig(f"{save_dir}/{i}_countplot.png", dpi=300)
    plt.close()


numeric_list = ['Age','BMI','Sleep Hours Per Day','Heart Attack Risk']
df_numeric = df.loc[:,numeric_list]
sns.pairplot(df_numeric,hue = "Heart Attack Risk", diag_kind="kde")
plt.show()
pairplot = sns.pairplot(df_numeric, hue="Heart Attack Risk", diag_kind="kde")
pairplot.savefig(f"{save_dir}/pairplot_numeric_features.png", dpi=300)

# %% EDA : box, swarm, cat, correlation analysis

save_dir = "/Users/pelinkonak/Desktop/adsız klasör/MakineOgrenmesiProjeleri/grafikler"
os.makedirs(save_dir, exist_ok=True)

#standart scaler

scaler = StandardScaler()
scaled_array=scaler.fit_transform(df[numeric_list[:-1]])

df_dummy = pd.DataFrame(scaled_array, columns=numeric_list[:-1])
df_dummy = pd.concat([df_dummy, df.loc[:, "Heart Attack Risk"]], axis=1)
    
#box plot
data_melted = pd.melt(df_dummy, id_vars="Heart Attack Risk", var_name = "features",value_name="value")

plt.figure(figsize=(10, 6))
sns.boxplot(x = "features", y="value", hue = "Heart Attack Risk", data = data_melted)
plt.title("Standard Scaled Feature Boxplot by Heart Attack Risk")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f"{save_dir}/eda_boxplot_scaled_features.png", dpi=300)
plt.close()

#swarm plot
plt.figure(figsize=(10, 6))
sns.boxplot(x = "features", y="value", hue = "Heart Attack Risk", data = data_melted)
plt.title("Standard Scaled Feature Swarmplot by Heart Attack Risk")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f"{save_dir}/eda_swarmplot_scaled_features.png", dpi=300)
plt.close()

#cat plot
catplot = sns.catplot(
    data=data_melted, x="features", y="value", hue="Heart Attack Risk", # Eğer cinsiyete göre ayrı ayrı grafikler istenirse
    kind="box",
    height=6,
    aspect=1.2
)
catplot.set_xticklabels(rotation=45)
catplot.fig.suptitle("Standard Scaled Feature Catplot by Heart Attack Risk", y=1.05)
catplot.tight_layout()
catplot.savefig(f"{save_dir}/eda_catplot_scaled_features.png", dpi=300)
plt.close()

#correlation
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".1f", linewidths=0.7, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{save_dir}/correlation_matrix.png", dpi=300)
plt.close()

# %% outlier detection

numeric_list = ['Age','BMI','Sleep Hours Per Day','Heart Attack Risk']
df_numeric = df.loc[:,numeric_list]

""" describe içinde
IQR = %75 - %25

UPPER BOUND = IQR * 2.5 + %75
LOWER BOUND = IQR * 2.5 + %25


"""

for i in numeric_list:
    
    #IQR
    Q1 = np.percentile(df.loc[:,i], 25)
    Q3 = np.percentile(df.loc[:,i], 75)
    
    IQR = Q3 - Q1
    
    print(f"old shape: {df.loc[:i].shape}")
    
    #upper band
    upper = np.where(df.loc[:,i] >= (Q3 + 2.5*IQR))
    
    #lower band
    lower = np.where(df.loc[:,i] <= (Q1 - 2.5*IQR))

    try:
        df.drop(upper[0], inplace = True)
    except: print("hata")

    
    try:
        df.drop(lower[0], inplace = True)
    except: print("hata")
    
    
    print(f"New shape: {df.shape}")
    
    """
    old shape: (8763, 26)
    New shape: (8763, 26)
    
    old shape: (8763, 26)
    New shape: (8763, 26)
    
    old shape: (8763, 26)
    New shape: (8763, 26)
    
    old shape: (8763, 26)
    New shape: (8763, 26)
    
    
    """

# %% modelling and hyperparameter tuning

# Örnek: Cholesterol sütunundaki string değerleri sayıya çevir
# Örnek: Cholesterol sütunundaki string değerleri sayıya çevir
df['Cholesterol'] = df['Cholesterol'].replace({'Low': 0, 'Average': 1, 'High': 2})

df['Blood Pressure'] = df['Blood Pressure'].replace({'Low': 0, 'Normal': 1, 'High': 2})
df['Diet'] = df['Diet'].replace({'Poor': 0, 'Average': 1, 'Good': 2})


df1 = df.drop(["Patient ID", "Country", "Continent", "Hemisphere"], axis=1)

#categorical to numeric one hot encoding

df1 = pd.get_dummies(df1, columns = categorical_list[:-1], drop_first= True)


X = df1.drop(["Heart Attack Risk"], axis =1)
y=df1[["Heart Attack Risk"]]

# Sayısal olmayan kalan sütunları kaldır veya dönüştür
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()
y = y.loc[X.index]  # y'yi hizala


#scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X[numeric_list[:-1]] = scaler.fit_transform(X[numeric_list[:-1]])


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_prob = logreg.predict_proba(X_test)
y_pred = (y_pred_prob[:, 1] >= 0.5).astype(int)  # 0.5 threshold ile sınıflandırma


print(f"test accuracy: {accuracy_score(y_test, y_pred)}")

#roc curve

fpr, tpr, threshold = roc_curve(y_test,y_pred_prob[:,1])

#plot curve
plt.plot([0,1],[0,1], "k--")
plt.plot(fpr, tpr, label = "logistic regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

save_dir = "/Users/pelinkonak/Desktop/adsız klasör/MakineOgrenmesiProjeleri/grafikler"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(f"{save_dir}/LogisticRegressionROCurve.png", dpi=300)
plt.close()


#hyperparameter tuning
lr = LogisticRegression()

penalty = ["l1","l2"]
parameters = {"penalty":penalty}

lr = LogisticRegression(solver='liblinear') 


lr_searcher = GridSearchCV(lr, parameters, cv=5)
lr_searcher.fit(X_train, y_train)

print(f"best parameters: {lr_searcher.best_params_}")
y_pred = lr_searcher.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test,y_pred)}")




