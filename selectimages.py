import random
import subprocess
import os

i=0
indices=[]
while True:
    if i==400:
        break
    index=random.randint(0,69999)
    if index not in indices:
        indices.append(index)
        i=i+1
        if i<=100:
            subprocess.check_output(['bash', '-c', "cp "+os.path.join("../datasets/ffhq_128x128/img","%05d.png" % index)+" ../datasets/ffhq_selected_1/"])
        elif i<=200:
            subprocess.check_output(['bash', '-c', "cp "+os.path.join("../datasets/ffhq_128x128/img","%05d.png" % index)+" ../datasets/ffhq_selected_2/"])
        elif i<=300:
            subprocess.check_output(['bash', '-c', "cp "+os.path.join("../datasets/ffhq_128x128/img","%05d.png" % index)+" ../datasets/ffhq_selected_3/"])
        else:
            subprocess.check_output(['bash', '-c', "cp "+os.path.join("../datasets/ffhq_128x128/img","%05d.png" % index)+" ../datasets/ffhq_selected_4/"])


        
