import torch
import torch.nn.functional as F
import torch.nn as nn


class DecisionModel(nn.Module):
    def __init__(self):
        super(DecisionModel, self).__init__()
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x1, x2):
        r1 = self.fc2(F.gelu(self.fc1(x1)))
        r2 = self.fc2(F.gelu(self.fc1(x2)))

        output = F.log_softmax(
            torch.concat((r1, r2), dim=1),
            dim=1
        )
        return output


class RewardModel(nn.Module):
    def __init__(self, enc_model, decision, device):
        super(RewardModel, self).__init__()
        self.enc_model = enc_model
        self.decision = decision
        self.device = device

    def forward(self, x):
        # x: [x1, x2, label]
        input_ids = x[0]['input_ids'].squeeze(dim=1).to(self.device)
        attention_mask = x[0]['attention_mask'].squeeze(dim=1).to(self.device)
        out1 = self.enc_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        out1 = out1.hidden_states[-1][:, -1, :]

        input_ids = x[1]['input_ids'].squeeze(dim=1).to(self.device)
        attention_mask = x[1]['attention_mask'].squeeze(dim=1).to(self.device)
        out2 = self.enc_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        out2 = out2.hidden_states[-1][:, -1, :]

        output = self.decision(out1, out2)
        label = x[-1].to(self.device)

        return output, label
