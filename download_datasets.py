
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


#raw data

## intercoder reliability files:

[wget.download(i, 'data/raw') for i in ["https://osf.io/5c6jz/download", "https://osf.io/p6qg3/download" ] ]

#file: kamervragen merged met annotated:
url1: "https://osf.io/9jdvy/downlaod"
wget.download(url1, 'data/raw')

#file: news merged met annotated:
url2  "https://osf.io/3m24v/download"
wget.download(url2, 'data/raw')

# file: only RPA coding (without content)
url3 : "https://osf.io/3ry4z/download"
wget.download(url3, 'data/raw')

# file with dictionary scores
url4 = "https://osf.io/gf6xy/download"
wget.download(url4, 'data/intermediate')


# word embedding file

url5 = "https://osf.io/ju9fd/download"
wget.download(url5, 'data/tmp')
