from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix , precision_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = load_iris()
x = data.data
y = data.target
# splitting data 
x_train , x_test  , y_train , y_test = train_test_split(x , y, test_size=0.2 , random_state=23)

# models
log = LogisticRegression()
svc = SVC()
tree = DecisionTreeClassifier()

#model
model = VotingClassifier(
    estimators= [('log regression' , LogisticRegression()) , 
                 ('Dtree classifier' , DecisionTreeClassifier()) ,
                 ('sv classifier' , SVC())] ,
    voting = 'hard' ,
    n_jobs = -1
)
# fitting & predicting  
model.fit(x_train , y_train)
predictions = model.predict(x_test)

# results 
voting_type = model.voting
estimators_name = model.named_estimators
n_feat = model.n_features_in_
accuracy_test = model.score(x_test , y_test) 
accuracy_train = model.score(x_train , y_train) 
confusion_matrix = confusion_matrix(y_test , predictions)
precision = precision_score(y_test , predictions , average='macro')

print("voting type :" , voting_type)
print("features number :" , n_feat)
print(f"precision = {precision*100:.2f}%")
print(f"accuracy (test) : {accuracy_test*100:.2f}%")
print(f"accuracy (train) : {accuracy_train*100:.2f}%")
print("confusion matrix : \n" , confusion_matrix)
print("estimators used : \n")
for key , value in estimators_name.items():
    print(f"{key} : {value}")

# plotting accuracy train & test
import matplotlib.pyplot as plt
accuracy_train = model.score(x_train, y_train)
accuracy_test = model.score(x_test, y_test)

# plotting
plt.style.use("fivethirtyeight")
plt.figure(figsize=(6,4))
plt.bar(['Train Accuracy', 'Test Accuracy'], [accuracy_train, accuracy_test], color=['orange', 'cyan'])
plt.ylabel('Accuracy')
plt.title('Comparison of Train and Test Accuracy')
plt.show()
