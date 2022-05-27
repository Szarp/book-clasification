import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords
import re


class Book:
    def __init__(
        self, remove_stopwords: bool = True, path: str = "data/booksummaries.txt"
    ) -> None:
        self.path = path
        self.df: pd.DataFrame = None
        self.remove_stopwords = remove_stopwords
        self.columns = ["Index", "Category", "Title", "Author", "Date", "Genres", "Summary"]
        self.sep = "\t"
        pass

    def load_csv(self, replace_nan: bool = True, have_header: bool = False):
        if not have_header:
            df = pd.read_csv(self.path, sep=self.sep)
            df.columns = self.columns
            if "Index" in df.columns:
                df.drop("Index", axis=1, inplace=True)
            if "Date" in df.columns:
                df.drop("Date", axis=1, inplace=True)
            if "Category" in df.columns:
                df.drop("Category", axis=1, inplace=True)
        else:
            df = pd.read_csv(self.path, sep=self.sep, header=0)
        if replace_nan:
            df = df.replace({np.nan: ""})
            df = df.replace({float("nan"): ""})
        self.df = df

    def clear_genres(self, many_genres: bool = True, one_genre_method: str = "first"):
        self.df = self.df.drop(self.df[self.df["Genres"] == ""].index)
        if many_genres:
            genres_cleaned = []
            for i in self.df["Genres"]:
                genres_cleaned.append(list(json.loads(i).values()))
            self.df["Genres"] = genres_cleaned
        else:
            genres_cleaned = []
            for i in self.df["Genres"]:
                if one_genre_method == "first":
                    genres_cleaned.append(list(json.loads(i).values())[0])
                else:
                    pass
            self.df["Genres"] = genres_cleaned

    def clear_summary(self, remove_stopwords: bool = True, remove_noneletters: bool = True):
        self.df = self.df.drop(self.df[self.df["Summary"] == ""].index)
        if remove_stopwords:
            summary_cleaned: list = []
            sw_nltk: list = stopwords.words("english")
            for record in self.df["Summary"]:
                words = [word for word in record.split() if word.lower() not in sw_nltk]
                summary_cleaned.append(" ".join(words))
            self.df["Summary"] = summary_cleaned
        if remove_noneletters:

            def clean_summary(text):
                # text = re.sub("\'s", "", text)
                text = re.sub("'", "", text)
                text = re.sub('"', "", text)
                text = re.sub("[^a-zA-Z]", " ", text)
                text = " ".join(text.split())
                text = text.lower()
                return text

            self.df["Summary"] = self.df["Summary"].apply(lambda x: clean_summary(x))

            pass

    def summary_statistics(
        self, count_words: bool = True, count_sentences: bool = True, count_chars: bool = True
    ):
        if count_words:
            self.df["words"] = pd.Series([len(summary.split()) for summary in self.df["Summary"]])
            pass
        if count_sentences:
            self.df["sentences"] = pd.Series(
                [len(summary.split(".")) for summary in self.df["Summary"]]
            )
            pass
        if count_chars:
            self.df["chars"] = pd.Series([len(summary) for summary in self.df["Summary"]])
            pass

    def save_data(self, path: str = "./data/booksummaries1.txt"):
        a: pd.DataFrame = self.df
        a.to_csv(path, sep=self.sep, index=False)
