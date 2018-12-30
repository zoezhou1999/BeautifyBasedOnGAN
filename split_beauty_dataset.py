import os
import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt

# initialize parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--csv', '-csv', help='path to csv file', default='./All_Ratings.csv', type=str)
args = parser.parse_args()

# 4 lists for each new csv
caucasian_male_rows =[]
caucasian_female_rows =[]
asian_male_rows =[]
asian_female_rows =[]

# read raters csv file
with open(args.csv, 'r') as csvfile:

    raw_dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    # split rows
    for i, row in enumerate(raw_dataset):
        row = ','.join(row)
        row = row.split(',')

        if (row[1])[:2].lower() == "cm":
            caucasian_male_rows.append('{0},{1},{2},'.format(row[0],row[1],row[2]))
        elif (row[1])[:2].lower() == "cf":
            caucasian_female_rows.append('{0},{1},{2},'.format(row[0],row[1],row[2]))
        elif (row[1])[:2].lower() == "am":
            asian_male_rows.append('{0},{1},{2},'.format(row[0],row[1],row[2]))
        else: # starts with AF
            asian_female_rows.append('{0},{1},{2},'.format(row[0],row[1],row[2]))

# write csv lines to files
dataset_path = os.path.dirname(args.csv)


csv_path = "{0}/splited/caucasian_male/All_Ratings.csv".format(dataset_path)
with open(csv_path, "wb") as csv_file:
    for line in caucasian_male_rows:
        csv_file.write(line)
        csv_file.write('\n')

csv_path = "{0}/splited/caucasian_female/All_Ratings.csv".format(dataset_path)
with open(csv_path, "wb") as csv_file:
    for line in caucasian_female_rows:
        csv_file.write(line)
        csv_file.write('\n')

csv_path = "{0}/splited/asian_male/All_Ratings.csv".format(dataset_path)
with open(csv_path, "wb") as csv_file:
    for line in asian_male_rows:
        csv_file.write(line)
        csv_file.write('\n')

csv_path = "{0}/splited/asian_female/All_Ratings.csv".format(dataset_path)
with open(csv_path, "wb") as csv_file:
    for line in asian_female_rows:
        csv_file.write(line)
        csv_file.write('\n')
