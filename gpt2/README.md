#

To train the model, do:

```bash
git clone https://github.com/souvikshanku/CALCS-2025.git

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd gpt2
python data/download_and_split.py

python train.py
```
