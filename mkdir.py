import pathlib 
import subprocess

# Create a directory
root_dir = pathlib.Path("/data04/junyi/ppmi/")
name_list = [path.name.split('_')[0] for path in root_dir.iterdir() if not path.name.endswith('.zip')]
name_set = set(name_list)
print(name_set)

for name in name_set:
    dir_name = root_dir / name
    dir_name.mkdir(parents=True, exist_ok=True)
    cmd = 'mv ' + str(root_dir) + f'/{name}' + '_* ' + str(dir_name)
    subprocess.run(cmd, shell=True)
    
    