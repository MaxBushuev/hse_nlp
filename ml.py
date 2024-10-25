import re
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.naive_bayes import GaussianNB
from cuml.ensemble import RandomForestClassifier
import cudf
from cuml.metrics import accuracy_score
from cuml.model_selection import GridSearchCV
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

def clean_data(elements):
    cleaned_elements = []
    for elem in elements:
        text = elem["text"]
        text = ' '.join(text.split())  # Remove extra spaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        cleaned_elements.append(elem)
    return cleaned_elements

def main():
    ds = load_dataset("dair-ai/emotion", "split")
    cleaned_train = clean_data(ds["train"])

    train_texts = [example['text'] for example in cleaned_train]
    val_texts = [example['text'] for example in ds['validation']]

    train_texts_series = cudf.Series(train_texts)
    val_texts_series = cudf.Series(val_texts)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(train_texts_series).todense()  # Converts to a dense matrix
    y_train = cudf.Series([example['label'] for example in cleaned_train])

    X_val_tfidf = vectorizer.transform(val_texts_series).todense()  # Converts to a dense matrix
    y_val = cudf.Series([example['label'] for example in ds['validation']])

    gnb = GaussianNB()
    gnb.fit(X_train_tfidf, y_train)

    y_val_pred = gnb.predict(X_val_tfidf)

    # Use cudf.Series directly without additional conversion
    print(f"Gaussian NB\nAccuracy: {accuracy_score(y_val, y_val_pred)}\nF1 Score: {f1_score(y_val.to_numpy(), y_val_pred.get(), average='weighted')}")

    rf = RandomForestClassifier(random_state=42, n_streams=1)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2],
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1_weighted')
    grid_search.fit(X_train_tfidf.get(), y_train.to_numpy())

    y_val_pred = grid_search.predict(X_val_tfidf.get())
    print(f"Grid search random forest\nAccuracy: {accuracy_score(y_val, y_val_pred)}\nF1 Score: {f1_score(y_val.to_numpy(), y_val_pred, average='weighted')}")

main()
