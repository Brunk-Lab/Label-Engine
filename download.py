import os
import gdown

# download datasets
save_folder ='data'
os.makedirs(f'{save_folder}/train', exist_ok=True)
os.makedirs(f'{save_folder}/val', exist_ok=True)
os.makedirs(f'{save_folder}/test', exist_ok=True)

# DOTA train data part 1
dota_data_url = 'https://drive.google.com/uc?id=1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2'
output = f'{save_folder}/train/part1.zip'
gdown.download(dota_data_url, output, quiet=False)

# DOTA train data part 2
dota_data_url = 'https://drive.google.com/uc?id=1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v'
output = f'{save_folder}/train/part2.zip'
gdown.download(dota_data_url, output, quiet=False)

# DOTA train data part 3
dota_data_url = 'https://drive.google.com/uc?id=1pEmwJtugIWhiwgBqOtplNUtTG2T454zn'
output = f'{save_folder}/train/part3.zip'
gdown.download(dota_data_url, output, quiet=False)

# DOTA val data
dota_data_url = 'https://drive.google.com/uc?id=1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP'
output = f'{save_folder}/val/part1.zip'
gdown.download(dota_data_url, output, quiet=False)

# DOTA test data part 1
dota_data_url = 'https://drive.google.com/uc?id=1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK'
output = f'{save_folder}/test/part1.zip'
gdown.download(dota_data_url, output, quiet=False)

# DOTA test data part 2
dota_data_url = 'https://drive.google.com/uc?id=1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv'
output = f'{save_folder}/test/part2.zip'
gdown.download(dota_data_url, output, quiet=False)