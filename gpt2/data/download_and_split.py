import pandas as pd


if __name__ == "__main__":
    original_df = pd.read_parquet(
        "hf://datasets/garrykuwanto/cspref/data/train-00000-of-00001.parquet"
    )

    unique_pairs = original_df[
        ['original_l1', 'original_l2']
    ].drop_duplicates()

    test_pairs = unique_pairs.sample(n=30, random_state=42)

    train_set = original_df[~original_df[
        ['original_l1', 'original_l2']].apply(tuple, axis=1).isin(
        test_pairs.apply(tuple, axis=1))
    ]

    test_set = original_df[original_df[
        ['original_l1', 'original_l2']].apply(tuple, axis=1).isin(
        test_pairs.apply(tuple, axis=1))
    ]

    train_set.to_csv("data/calcs_train_split.csv", index=None)
    test_set.to_csv("data/calcs_test_split.csv", index=None)

    print(f"Train set shape: {train_set.shape}")
    print(f"Test set shape: {test_set.shape}")
