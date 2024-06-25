import itertools
import subprocess
import json

lr = [1e-3, 1e-4, 1e-5]
depth = [3, 5],
embed_size = [256, 512]

config_path = 'config/default.json'

print(len(list(itertools.product(lr, depth, embed_size))))

for i, (n, g, k) in enumerate(itertools.product(lr, depth, embed_size)):
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['lr'] = n
    data['depth'] = g
    data['embed_size'] = k
    data["grid_num"] = i
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'train.py', '--config', config_path])
    process.wait()
