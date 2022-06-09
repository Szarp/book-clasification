import nltk
from nltk.corpus import stopwords
from regex import B
from dataset import Dataset

nltk.data.path.append("./nltk")

if __name__ == "__main__":
    print("Hello world!")
    # b: Dataset = Dataset(path="data/booksummaries1.txt")

    # Example for four generes given from eportal
    b: Dataset = Dataset()
    b.load_csv()
    b.clear_genres()
    b.clear_summary()
    b.summary_statistics()
    b.df = b.df[b.df["Genres"].map(len) == 1]
    b.save_data(path="data/books_four_genres.txt")
