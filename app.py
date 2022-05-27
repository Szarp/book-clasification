import nltk
from nltk.corpus import stopwords
from regex import B
from dataset import Book
nltk.data.path.append("./nltk")

if __name__ == "__main__":
    print("Hello world!")
    b: Book = Book(path="data/booksummaries1.txt")
    # b:Book = Book()
    b.load_csv(have_header=True)
    # b.clear_genres()
    # b.clear_summary()
    # b.summary_statistics()
    # b.save_data()
    # df:pd.DataFrame = load_dataset()
    # df.columns = ["index","category","Title","Authot","Date","Genre","Summary"]
    print(b.df["chars"])
