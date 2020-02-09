import random
import subprocess

i=0
indices=[]
while True:
    if i==400:
        break
    index=random.randint(0,69999)
    if index not in indices:
        indices.append(index)
        subprocess.check_output(['bash', '-c', "cp "+os.path.join("./datasets/ffhq","%05d.png" % index)+" ./datasets/ffhq_selected/"])
        i=i+1
