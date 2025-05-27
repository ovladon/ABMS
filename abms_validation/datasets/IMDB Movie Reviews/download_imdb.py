import urllib.request
import tarfile
import os

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
urllib.request.urlretrieve(url, "aclImdb_v1.tar.gz")

with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
    tar.extractall(".")

print("IMDB dataset downloaded and extracted")
