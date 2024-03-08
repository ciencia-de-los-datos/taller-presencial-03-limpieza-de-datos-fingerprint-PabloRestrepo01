"""Taller evaluable presencial"""

import nltk
import pandas as pd


def load_data(input_file):

    data = pd.read_csv(input_file, sep="\t")
    return data



def create_fingerprint(df):

    df = df.copy()
    df["fingerprint"] = df["text"]
    df["fingerprint"] = (
        df["fingerprint"]
        .str.strip()
        .str.lower()
        .str.replace("-","")
        .str.replace(".","")
        .str.split()
        .apply(lambda x: [nltk.PorterStemmer().stem(w) for w in x])
        .apply(lambda x: sorted(set(x)))
        .str.join(" ")
    )
    return df


def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""

    df = df.copy()
    fingerprints = df.sort_values(by = ["fingerprint", "text"]).copy()
    fingerprints = df.groupby("fingerprint").first().reset_index()
    fingerprints = fingerprints.set_index("fingerprint")["text"].to_dict()
    df["cleaned"] = df["fingerprint"].map(fingerprints)
    return df


def save_data(df, output_file):
    
    df = df.copy()
    df = df[["cleaned"]]
    df = df.rename(columns = {"cleaned": "text"})
    df.to_csv(output_file, sep = "\t", index = False)


def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""

    df = load_data(input_file)
    df = create_fingerprint(df)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )