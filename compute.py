from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from distances import euclidean_distance, manhattan_distance, chebyshev_distance
from knn import KNN

def load_data():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

def split_data(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def evaluate(X_train, X_test, y_train, y_test, k, distance_fn):
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test, distance_fn)
    
    accuracy = np.mean(y_pred == y_test)
    return accuracy
