import sys
sys.stdout.reconfigure(encoding='utf-8')
from datasets import load_dataset

# Check VietTung04/alqac2025-reasoning-trace
try:
    ds = load_dataset('VietTung04/alqac2025-reasoning-trace')
    print('alqac2025-reasoning-trace splits:', list(ds.keys()))
    for s in sorted(ds.keys()):
        print(f'  {s}: {len(ds[s])}')
        if len(ds[s]) > 0:
            print(f'    Keys: {list(ds[s][0].keys())}')
except Exception as e:
    print(f'alqac2025-reasoning-trace: {e}')

# Check VMTEB-ALQAC-retrieval-wseg
try:
    ds2 = load_dataset('another-symato/VMTEB-ALQAC-retrieval-wseg')
    print('\nVMTEB-ALQAC-retrieval-wseg splits:', list(ds2.keys()))
    for s in sorted(ds2.keys()):
        print(f'  {s}: {len(ds2[s])}')
except Exception as e:
    print(f'\nVMTEB-ALQAC-retrieval-wseg: {e}')
