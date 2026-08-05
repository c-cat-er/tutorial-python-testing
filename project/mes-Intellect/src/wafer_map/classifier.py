from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def classify_defects(features, labels=None):
    if labels is None:
        # 規則式分類（專案三）
        return ["Ring" if f[0] > 0.3 else "Random" for f in features]
    # ML 分類（專案四）
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    return clf.predict(features), accuracy_score(y_test, clf.predict(X_test))