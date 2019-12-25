import os
import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt

# initialize parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--csv', '-csv', help='path to csv file', default='../All_Ratings.csv', type=str)
parser.add_argument('--density', '-density', help='configure plot density', default=0.05, type=float)
args = parser.parse_args()

# initiate list of beauty rates means
beauty_rates_mean = []

# read raters csv file
with open(args.csv, 'r') as csvfile:
    
    # Dictionary to load images from csv
    # key: image name
    # value: list of 60 beauty rates from raters
    csv_dict = {}
    
    raw_dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    # fill the dictionary
    for i, row in enumerate(raw_dataset):
        row = ','.join(row)
        row = row.split(',')
            
        # create list of rates for each image
        if row[1] in csv_dict:
            csv_dict[row[1]][0].append(float(row[2]))
        else:
            csv_dict[row[1]] = [[float(row[2])]]

    # move dict to lists, convert beauty rates to numpy ranged in [0,1]
    for key, value in csv_dict.items():
        beauty_rates_mean.append(np.mean(np.asarray(value, dtype=np.float32)))

# create a x axis with the given density and zeros as y axis to be filled next
x_values = np.arange(0.0, 5.0, args.density)
y_values = [0]*len(x_values)

# for each mean, increase the counter in the correct location
for val in beauty_rates_mean:
    y_values[int(round(val/args.density))] += 1

# plot the results
plt.plot(x_values, y_values)
plt.xlabel('beauty rates')
plt.ylabel('number of subjects')
plt.title('Beauty Rates Distribution')
plt.grid(True)
plt.savefig(os.path.basename(args.csv).split(".")[-2]+ ".png")

