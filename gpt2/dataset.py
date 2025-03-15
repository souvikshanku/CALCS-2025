import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.tokenized_data = []
        self.label_dict = {
            0: torch.tensor([1, 0]),
            1: torch.tensor([0, 1]),
        }

        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "return_attention_mask": True,
            "return_tensors": "pt"
        }

        for idx in range(len(df)):
            prepend = (
                df.iloc[idx]["original_l1"]
                + "+"
                + df.iloc[idx]["original_l2"]
                + "->"
            )
            text1 = prepend + df.iloc[idx]["sent_1"]
            text2 = prepend + df.iloc[idx]["sent_2"]
            text1_enc = self.tokenizer.encode_plus(text1, **kwargs)
            text2_enc = self.tokenizer.encode_plus(text2, **kwargs)

            chosen = df.iloc[idx]["chosen"]
            if chosen in [0, 1]:
                label = self.label_dict[chosen]
            # in case of a tie, randomly assign one as chosen
            elif chosen == 2:
                label = torch.randperm(2)

            self.tokenized_data.append((text1_enc, text2_enc, chosen, label))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.tokenized_data[item]
