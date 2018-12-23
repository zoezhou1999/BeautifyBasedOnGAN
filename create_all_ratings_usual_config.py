import os
import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pdb

def create_all_ratings_csv_usual_config(series_name = 'a', ratings = 200, new_csv_name = 'all_ratings_series_'): ## series name is a/b (for non-smiles / smile)



    # Dictionary to load images from csv
    # key: rater's birth date
    # value: 200 given beauty rates
    csv_dict = {}

    # read raters csv file

    ratings_file = 'Rate_Facial_Beauty.csv'
    with open(new_csv_name + series_name + '.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')

        with open(ratings_file, 'r', encoding="latin1") as csvfile:
            raw_dataset = csv.reader(csvfile, delimiter=',', quotechar='|')

            # fill the dictionary
            for i, row in enumerate(raw_dataset):

                if i == 0:  # skip the first row (only titles)
                    continue

                row = ','.join(row)
                row = row.split(',')

                # initiate a list with the birth date as key
                csv_dict[row[1]] = []
                for j in range(13, 13 + ratings):
                    csv_dict[row[1]].append(int(row[j][1]))

                # create numpy array of the csv
                ratings_np = np.array(csv_dict[row[1]])
                for index in range(len(ratings_np)):
                    im_name = str(index + 1) + series_name +'.jpg'
                    writer.writerow([i,im_name,ratings_np[index]])

def main():
    series_name = 'a'
    create_all_ratings_csv_usual_config(series_name)

if __name__ =='__main__':
    main()
