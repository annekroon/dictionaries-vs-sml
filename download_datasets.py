
import wget
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

dirNames = ['data/' , 'data/intermediate' , 'data/raw' , 'data/tmp']

for i in dirNames:
    try:
        os.mkdir(i)
        print("Directory " , i ,  " Created ")
    except FileExistsError:
        print("Directory " , i ,  " already exists")

# intermediate dataset

# file with dictionary scores
url1 = "https://osf.io/gf6xy/download"
wget.download(url1, 'data/intermediate')

#raw data

#file: kamervragen merged met annotated:
url3: "https://osf.io/9jdvy/downlaod"
wget.download(url3, 'data/raw')

#file: news merged met annotated:
url4  "https://osf.io/3m24v/download"
wget.download(url4, 'data/raw')

# file: only RPA coding (without content)
url5 : "https://osf.io/3ry4z/download"
wget.download(url5, 'data/raw')


# word embedding file

url2 = "https://osf.io/ju9fd/download"
wget.download(url1, 'data/tmp')
