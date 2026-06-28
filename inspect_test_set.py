import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
data = json.load(open('test_set.json', encoding='utf-8'))
print(f"Total questions: {len(data)}")
for q in data[:30]:
    print(f"{q['id']}: {q['question']}")
