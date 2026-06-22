import os
import os.path as osp
import glob
import argparse


def main(args):
    vol_dataset_path = args.vol
    output_path = args.output
    scanner_path = args.scanner
    projections_num = args.projections_num
    device = args.device

    vol_file_paths = sorted(glob.glob(osp.join(vol_dataset_path, "*.npy")))

    if len(vol_file_paths) == 0:
        raise ValueError("{} find no *.npy file!".format(vol_file_paths))

    for vol_file_path in vol_file_paths:
        cmd = f"CUDA_VISIBLE_DEVICES={device} python /home/haowei_zhou/Project/Gaussian_splatting/TMI/data_process/generate_data.py --vol {vol_file_path} --scanner {scanner_path} --output {output_path} --projections_num {projections_num}"
        os.system(cmd)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol", default="/home/haowei_zhou/Project/Gaussian_splatting/TMI/data/raw_volum", type=str, help="Path to vol dataset.")
    parser.add_argument("--scanner", default="/home/haowei_zhou/Project/Gaussian_splatting/TMI/data_process/scanner/cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/cone_projections", type=str, help="Path to output.")
    parser.add_argument("--projections_num", default=150, type=int, help="Number of projections to generate.")
    parser.add_argument("--device", default=0, type=int, help="GPU device.")
    # fmt: on

    args = parser.parse_args()
    main(args)
# python /home/haowei_zhou/Project/Gaussian_splatting/TMI/data_process/generate_data_all.py --vol /home/haowei_zhou/Project/Gaussian_splatting/TMI/data/raw_volum --scanner /home/haowei_zhou/Project/Gaussian_splatting/TMI/data_process/scanner/cone_beam.yml --output data/projs_128