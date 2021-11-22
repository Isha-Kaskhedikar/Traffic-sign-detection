import os
from PIL import Image 
from pathlib import Path

directory = r'E:/sem V/DL and FL/acc-german-traffic-sign-classification/GTSRB_Challenge/train'
ds=r'E:\sem V\DL and FL\train2'
for filename in os.listdir(directory):
    m=os.path.join(ds,filename)
    Path(m).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(os.path.join(directory, filename)):        
        n=os.path.join(directory, filename,f)
        print(n)
        im = Image.open(n)
        p=os.path.join(m,f)
        im.convert('RGB').save(p+".jpg","JPEG") #this converts png image as jpeg
    print("---"*30)
    