import csv
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='./results.csv', help='path to csv file')
parser.add_argument('--seq', type=int, default=10, help='number of sequences')
parser.add_argument('--imgs', type=int, default=3, help='number of images in each sequence')
parser.add_argument('--parallel', dest='parallel', action='store_true')
opt = parser.parse_args()
print(opt)

# initiate loss
loss = 0

# read results csv file
with open(opt.csv, 'r') as csvfile:
    
    raw_dataset = csv.reader(csvfile, delimiter=',', quotechar='|')
    # calculate number of valid answers to compute mean later
    num_of_lines = 0
    
    for i, row in enumerate(raw_dataset):
        
        # skip titles row
        if i == 0:
            continue
    
        row = ','.join(row)
        row = row.split(',')
        
        # assume each user is valid until we check it
        is_user_valid = True
        num_of_lines += 1
        
        # initiate loss for user
        seq_loss = 0
        
        # compute mean squared loss
        for j in range(opt.seq):
            
            # list to check that the user understood the mission
            found = [0,0,0]
            for k in range(opt.imgs):
                
                # mark the choice of the user and compute loss
                found[(int(row[3+j*opt.imgs+k][1:-1]))-1] = 1
                seq_loss += (int(row[3+j*opt.imgs+k][1:-1]) - (k+1))**2
            
            if 0 in found: # the user failed to understand the mission so we put him out
                is_user_valid = False

        # add the loss if the user is valid, otherwise remove him
        if is_user_valid:
            loss += seq_loss / (opt.imgs * opt.seq)
        else:
            num_of_lines -= 1
    
    # compute average of losses from users
    loss /= num_of_lines - 1

print("Mean Squared Error: {}".format(loss))
print("Calculated from {} valid answers".format(num_of_lines))




