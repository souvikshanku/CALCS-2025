import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from transformers import AutoTokenizer, AutoModel

from dataset import TextDataset
from model import DecisionModel, RewardModel


def preprocess(dataset_path: str, sample_frac: float = 1) -> pd.DataFrame:
    df_train = pd.read_csv(dataset_path)
    chosen_class = {
        "sent_1": 0,
        "sent_2": 1,
        "tie": 2
    }
    df_train["chosen"] = df_train["chosen"].apply(lambda x: chosen_class[x])
    return df_train.sample(frac=sample_frac)


def load(model: str, df: pd.DataFrame, batch_size: int = 8) -> DataLoader:
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextDataset(df, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_dataloader


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = 'gpt2'

    df = preprocess("data/calcs_train_split.csv", 0.1)
    train_dataloader = load(model, df, batch_size=14)

    enc_model = AutoModel.from_pretrained("gpt2")
    dm = DecisionModel()
    rm = RewardModel(enc_model, dm, device)
    rm.to(device)

    optimizer = torch.optim.AdamW([
        {"params": rm.decision.parameters(), "lr": 3e-5},
        {"params": rm.enc_model.parameters(), "lr": 3e-5},
    ], lr=3e-5)

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    epochs = 5
    accumulation_steps = 2

    for epoch in range(epochs):
        rm.train()
        total_loss = 0

        for batch_id, batch in enumerate(train_dataloader):
            out, label = rm(batch)
            loss = - (out * label).sum() / label.shape[0]
            loss.backward()

            if (batch_id + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # if batch_id % 200 == 0:
            print(loss.item())

        # If the number of batches isn't divisible by accumulation_steps,
        # do one final step after the loop ends
        if (batch_id + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        avg_loss = total_loss / len(train_dataloader)

        print(f"Epoch {epoch + 1}")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print("\n")

        checkpoint = {
            'model_state_dict': rm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'avg_loss': avg_loss
        }

        torch.save(checkpoint, f"ckpt/gpt2_rm_epoch{epoch + 1}.pth")
