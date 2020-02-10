from image_quality_metric.Python.libsvm.python.brisquequality import test_measure_BRISQUE
import argparse
import csv
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', '-results_dir', help='batch image beautification results', default='dean_cond_batch16', type=str)
parser.add_argument('--src_dir', '-src_dir', help='original images path', default='dean_cond_batch16', type=str)
parser.add_argument('--final_iteration', '-final_iteration', help='mark the final beautificaton result', default=572, type=int)
parser.add_argument('--csv_name', '-csv_name', help='csv file name', default='dean_cond_batch16', type=str)

args = parser.parse_args()

paths=sorted(glob.glob(os.path.join(args.src_dir,"*.png")))
with open(args.csv_name+ ".csv", mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['image_name', 'ori_qualityscore', 'res_qualityscore'])
    for path in paths:
        name=os.path.basename(path)
        name=name[0:name.find(".")]
        result_path=os.path.join(path,str(name))
        result_path_image=os.path.join(result_path,str(args.final_iteration)+"_0.png")
        # calculate quality score
        ori_qualityscore = test_measure_BRISQUE(path)
        res_qualityscore = test_measure_BRISQUE(result_path_image)
        writer.writerow([name, ori_qualityscore, res_qualityscore])