import argparse, os
from AI_worker import ai_worker
from Canny_worker import canny_worker

def main():
    parser = argparse.ArgumentParser(description="vessel contour detection")
    parser.add_argument('--p', type=str, help='path to image')
    parser.add_argument('--n', type=str, help='name of output mask')
    parser.add_argument('--t', type=str, help='type of algorithm (1 is Unet, 0 is Canny)')
    args = parser.parse_args()
    img_path = args.p
    name = args.n
    type = int(args.t)

    if not os.path.exists('output'):
        os.mkdir('output')

    if type == 1:
        worker = ai_worker(img_path, name)
        worker.seg()
    elif type == 0:
        worker = canny_worker(img_path, name)
        worker.seg()
    else:
        print('Wrong type code! Choose between 1 (Unet) and 0 (Canny)')

if __name__ == "__main__":
    main()