import sys
sys.stdout.reconfigure(encoding='utf-8')
from huggingface_hub import HfApi
api = HfApi()
results = api.list_datasets(search='alqac')
for r in results:
    print(f'{r.id}: gated={r.gated}, private={r.private}')
