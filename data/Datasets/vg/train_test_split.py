import json
import random

with open('vg_splits.json') as f:
  data = json.load(f)

print(len(data['train']))
print(len(data['test']))
print(len(data['val']))

all_data = data['train'] + data['test'] + data['val']

random.shuffle(all_data)

train_data = all_data[:len(data['train'])]
test_data=all_data[len(data['train']):len(data['train'])+len(data['test'])]
val_data=all_data[len(data['train'])+len(data['test']):]

assert len(data['train']) == len(train_data)
assert len(data['test']) == len(test_data)
assert len(data['test']) == len(val_data)

new_shuffled_dict = {'train': train_data, 'test':test_data, 'val':val_data}

f.close()

with open('vg_splits.json', 'w') as f:
  json.dump(new_shuffled_dict, f)

