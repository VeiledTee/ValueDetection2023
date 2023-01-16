import pandas as pd


def export_results(cols: list, arguments: list, results: list, filename: str):

    df = pd.DataFrame(columns=cols)
    df[df.columns[0]] = arguments

    for col, result in zip(cols[1:], results):
        df[col] = result
    print(df.head())
    df.to_csv(filename, sep='\t', index=False)
