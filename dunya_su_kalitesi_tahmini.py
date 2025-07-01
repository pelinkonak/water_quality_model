#keşifsel veri analizi EDA
#preprocessing : missing value problem, train-test split, normalization
#modelling: decision tree and random forest
#evaluation: decision tree visualization
#hyperparameter tuning: random forest

import numpy as np #linear algebra
import pandas as pd #dataprocessing
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization
import plotly.express as px #visualization

import missingno as msno #missing value analysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import precision_score, confusion_matrix

from sklearn import tree

df = pd.read_csv("water_potability.csv")

describe = df.describe()

df.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3276 entries, 0 to 3275
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   ph               2785 non-null   float64
 1   Hardness         3276 non-null   float64
 2   Solids           3276 non-null   float64
 3   Chloramines      3276 non-null   float64
 4   Sulfate          2495 non-null   float64
 5   Conductivity     3276 non-null   float64
 6   Organic_carbon   3276 non-null   float64
 7   Trihalomethanes  3114 non-null   float64
 8   Turbidity        3276 non-null   float64
 9   Potability       3276 non-null   int64  
dtypes: float64(9), int64(1)
memory usage: 256.1 KB
"""

# dependent variable analysis 
# Potability sınıf sayımlarını al, index'i sıfırla ve sütunları yeniden adlandır
d = df["Potability"].value_counts().reset_index()
d.columns = ["Potability", "count"]

# Pie chart
fig = px.pie(
    d,
    values="count",
    names="Potability",
    hole=0.35,
    opacity=0.8
)

# Etiketleri güncelle
fig.update_layout(
    title=dict(text="Pie Chart of Potability Feature"),
)
fig.update_traces(textposition="outside", textinfo="percent+label")

fig.show()
# %%
fig.write_html("potability_pie_chart.html")


# korelasyon analizi (clustermap düzeltildi ve kaydedildi)
g = sns.clustermap(df.corr(), cmap="coolwarm", dendrogram_ratio=(0.1, 0.2), annot=True, linewidths=0.8, figsize=(10, 10))
g.figure.savefig("correlation_clustermap.png", dpi=300)

#distribution of features

non_potable = df.query("Potability == 0")
potable = df.query("Potability == 1")

plt.figure()
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3,3 ,ax+1)
    plt.title(col)
    sns.kdeplot(x = non_potable[col], label = "Non Potable")
    sns.kdeplot(x =potable[col],label ="Potable")
    plt.legend()
    
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=300)
plt.show()

#missing value
msno.matrix(df)
plt.savefig("missingvalue.png", dpi=300)
plt.show()

#preprocessing

print(df.isnull().sum())
df["ph"].fillna(value = df["ph"].mean(),inplace = True)
df["Sulfate"].fillna(value = df["Sulfate"].mean(),inplace = True)
df["Trihalomethanes"].fillna(value = df["Trihalomethanes"].mean(),inplace = True)
  
print(df.isnull().sum())  

#train test split
X= df.drop("Potability",axis=1).values #independent değerler
y= df["Potability"].values #target değeri potable or non-potable
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


#min-max normalization 0-1
x_train_max = np.max(X_train)
x_train_min = np.min(X_train)
X_train = (X_train - x_train_min) / (x_train_max - x_train_min)
X_test = (X_test - x_train_min) / (x_train_max - x_train_min)


#modelling : decision tree and random forest

models = [("DTC", DecisionTreeClassifier(max_depth=3)),
          ("RF", RandomForestClassifier())]

finalResult = [] #score list
cmList = [] #confusion matrix list
for name, model in models:
    model.fit(X_train,y_train) #training
    
    model_result = model.predict(X_test) #prediction
    
    score = precision_score(y_test, model_result)
    
    finalResult.append((name,score))

    cm = confusion_matrix(y_test, model_result)
    cmList.append((name, cm))
    
print(finalResult)    

for name, cm in cmList:
    plt.figure()
    sns.heatmap(cm, annot = True, linewidths=0.8, fmt=" .0f")
    plt.title(name)
    plt.savefig(f"/Users/pelinkonak/Desktop/confusion_matrix_{name}.png", dpi=300)
    plt.show()

#Evaluation: decision tree visualization
dt_clf = models[0][1]

plt.figure()
tree.plot_tree(dt_clf,feature_names=df.columns.tolist()[:-1],
               class_names = ["0","1"],
               filled = True,
               precision = 5)
plt.savefig("/Users/pelinkonak/Desktop/decision_tree_plot.png", dpi=300)

plt.show()    

#Hyperparameter tuning : random forest

model_params = {
    "Random Forest":
        {
            "model":RandomForestClassifier(),
            "params":
                {
                    "n_estimators": [10,50,100],
                    "max_features": ["auto","sqrt","log2"],
                    "max_depth": list(range(1,21,3))
                     }
             }
 
     }
        
cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats=2)
scores= []

for model_name, params in model_params.items():
    
    rs = RandomizedSearchCV(params["model"], params["params"], cv = cv, n_iter =10)
    rs.fit(X,y)
    scores.append([model_name, dict(rs.best_params_),rs.best_score_])

print(scores)
       

[['Random Forest', {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': 13}, np.float64(0.6715490597654068)]]



