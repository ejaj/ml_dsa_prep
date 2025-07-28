import pandas as pd
from nlp_preprocessor import TextPreprocessor


def main():
    df = pd.read_csv('support_messages.csv')
    preprocessor = TextPreprocessor(language="en", lemmatize=True)
    preprocessed_df = preprocessor.transform(df)
    preprocessed_df.to_parquet('support_messages.parquet', index=False)


if __name__ == "__main__":
    main()
