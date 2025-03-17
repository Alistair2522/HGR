import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score

# DATA
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# MODELS

models = {  "decision_tree_gini" : DecisionTreeClassifier(criterion="gini"),
            "decision_tree_entropy" : DecisionTreeClassifier(criterion="entropy"),
            "decision_tree_log_loss" : DecisionTreeClassifier(criterion="log_loss"),
            "random_forest_gini" : RandomForestClassifier(criterion="gini"),
            "random_forest_entropy" : RandomForestClassifier(criterion="entropy"),
            "random_forest_log_loss" : RandomForestClassifier(criterion="log_loss"),
            "k_neighbors_n50" : KNeighborsClassifier(n_neighbors=50),
            "k_neighbors_n100" : KNeighborsClassifier(n_neighbors=100)
}
y_predict = {}
score = {}
best = "decision_tree_gini"
for model in models:
    models[model].fit(x_train, y_train)
    y_predict[model] = models[model].predict(x_test)
    score[model] = [round(100*accuracy_score(y_predict[model], y_test), 3),
                    round(100*precision_score(y_predict[model], y_test, average='weighted'), 3),
                    round(100*recall_score(y_predict[model], y_test, average='weighted'), 3)
    ]
    if score[model]>score[best]: best = model

print(score)

f = open('model.p', 'wb')
pickle.dump({'model': models[best]}, f)
f.close()
