import os
import shutil

classes = os.listdir('data/archive/EuroSat')
path_target = 'data/targets'
path_input = 'data/inputs'

if not os.path.exists(path_input):
  os.mkdir(path_input)
  os.mkdir(path_target)

k = 1
for kind in classes:
  path = os.path.join('data/archive/EuroSat', str(kind))
  if os.path.isfile(path):
    continue
  for i, f in enumerate(os.listdir(path)):
    shutil.copyfile(os.path.join(path, f),
                  os.path.join(path_target, f))
    os.rename(os.path.join(path_target, f), os.path.join(path_target, f'IMG_{k}.jpg'))
    k += 1

print("Process N = ", k)