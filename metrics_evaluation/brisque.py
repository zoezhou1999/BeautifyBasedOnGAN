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
mean_ori_qualityscore=0
mean_res_qualityscore=0
with open(args.csv_name+ ".csv", mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['image_name', 'ori_qualityscore', 'res_qualityscore'])
    for path in paths:
        name=os.path.basename(path)
        # name=name[0:name.find("_")]
        name=name[0:name.find(".")]
        #These for Beholder-XXXX
        result_path=os.path.join(args.results_dir,str(name))
        result_path_image=os.path.join(result_path,str(args.final_iteration)+"_0.png")
        # result_path_image=os.path.join(result_path,"%04d-0.png" % args.final_iteration)

        #These for InterFaceGAN-XXXX
        # result_path_image=os.path.join(args.results_dir,name+"_0.png")

        print(path,name,result_path_image)
        # print(path,name,result_path,result_path_image)
        # calculate quality score
        ori_qualityscore = test_measure_BRISQUE(path)
        res_qualityscore = test_measure_BRISQUE(result_path_image)
        mean_ori_qualityscore+=ori_qualityscore
        mean_res_qualityscore+=res_qualityscore
        writer.writerow([name, ori_qualityscore, res_qualityscore])

with open(args.csv_name+ ".txt", mode='w') as f:
    mean_ori_qualityscore=mean_ori_qualityscore/len(paths)
    mean_res_qualityscore=mean_res_qualityscore/len(paths)
    f.writelines("image num: {};\n".format(len(paths)))
    f.writelines("mean_ori_qualityscore: {};\n".format(mean_ori_qualityscore))
    f.writelines("mean_res_qualityscore: {};\n".format(mean_res_qualityscore))