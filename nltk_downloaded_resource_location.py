import nltk

print("NLTK package location:")
print(nltk.__file__)

print("\nNLTK data search paths:")
for path in nltk.data.path:
    print(path)

print("\nDefault NLTK download directory:")
print(nltk.downloader.Downloader().default_download_dir())