import re

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def clean_data(elements):
    cleaned_elements = []
    for elem in elements:
        text = elem["text"]
        # Remove extra spaces
        text = ' '.join(text.split())
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        cleaned_elements.append(elem)
    return cleaned_elements


def main():
    ds = load_dataset("dair-ai/emotion", "split")
    cleaned_train = clean_data(ds["train"])

    train_texts = [example['text'] for example in cleaned_train]
    val_texts = [example['text'] for example in ds['validation']]

    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(train_texts).toarray()
    y_train = [example['label'] for example in cleaned_train]

    X_val_tfidf = vectorizer.transform(val_texts).toarray()
    y_val = [example['label'] for example in ds['validation']]

    gnb = GaussianNB()
    gnb.fit(X_train_tfidf, y_train)

    y_val_pred = gnb.predict(X_val_tfidf)
    print(f"Gaussian NB\n{classification_report(y_val, y_val_pred)}")


    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1_weighted')
    grid_search.fit(X_train_tfidf, y_train)

    y_val_pred = grid_search.predict(X_val_tfidf)
    print(f"Grid search random forest\n{classification_report(y_val, y_val_pred)}")


main()