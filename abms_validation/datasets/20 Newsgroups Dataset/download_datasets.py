from sklearn.datasets import fetch_20newsgroups
import pickle

# Download all categories
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Save to files
with open('newsgroups_train.pkl', 'wb') as f:
    pickle.dump(newsgroups_train, f)
with open('newsgroups_test.pkl', 'wb') as f:
    pickle.dump(newsgroups_test, f)

print(f"Training samples: {len(newsgroups_train.data)}")
print(f"Test samples: {len(newsgroups_test.data)}")
print(f"Categories: {newsgroups_train.target_names}")
