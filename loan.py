import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV



# read the dataset
data = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
print(data.head())

print('\n\nColumn Names\n\n')
print(data.columns)

#label encode the target variable
encode = LabelEncoder()
data.Loan_Status = encode.fit_transform(data.Loan_Status)

# drop the null values
data.dropna(how='any',inplace=True)


# train-test-split
train , test = train_test_split(data,test_size=0.2,random_state=0)



# seperate the target and independent variable
train_x = train.drop(columns=['Loan_ID','Loan_Status'],axis=1)
train_y = train['Loan_Status']

test_x = test.drop(columns=['Loan_ID','Loan_Status'],axis=1)
test_y = test['Loan_Status']

# encode the data
train_x = pd.get_dummies(train_x)
test_x  = pd.get_dummies(test_x)

param_grid = [
    {'max_depth': [2,3,4], 'min_samples_split': [ 10, 20, 30]},
    {'max_depth': [2,3,4], 'min_samples_split': [ 10, 20, 30], 'criterion':['entropy']}
]

tree = DecisionTreeClassifier()

grid_search = GridSearchCV(tree, param_grid, cv=5)

grid_search.fit(train_x,train_y)

best_tree = grid_search.best_estimator_.fit(train_x,train_y)

predict = best_tree.predict(test_x)

print('Predicted Values on Test Data',predict)

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y,predict))
