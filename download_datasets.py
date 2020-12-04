
import wget
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
dirNames = ['data/' ]

for i in dirNames:
    try:
        os.mkdir(i)
        print("Directory " , i ,  " Created ")
    except FileExistsError:
        print("Directory " , i ,  " already exists")

all_data = 'https://surfdrive.surf.nl/files/index.php/s/m8QF1YC0jaY9SQM/download'
wget.download(all_data, 'data')
