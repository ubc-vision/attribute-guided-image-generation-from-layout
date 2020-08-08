import argparse
import os
from IPython import embed
from util import util
import models.dist_model as dm
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir0', type=str, default='~/checkpoints/all/results/sampled_att64_est_change_att_vg_bs12e64z64clstm3li1.0lo1.0lc1.0lz8.0lc1.0lk0.01/1')
parser.add_argument('--dir1', type=str, default='~/checkpoints/all/results/sampled_att64_est_change_att_vg_bs12e64z64clstm3li1.0lo1.0lc1.0lz8.0lc1.0lk0.01/2')
parser.add_argument('--out', type=str, default='~/checkpoints/all/results/layout2im_coco_dists12.txt')

parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
opt = parser.parse_args()

## Initializing the model
model = dm.DistModel()
model.initialize(model='net-lin', net='alex', use_gpu=opt.use_gpu)

# crawl directories
f = open(opt.out, 'w')
files = os.listdir(opt.dir0)

print(files)

all_dist01 = []
for file in files:
    if (os.path.exists(os.path.join(opt.dir1, file))):
        print("a")
        # Load images
        img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0, file)))  # RGB image from [-1,1]
        img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1, file)))

        # Compute distance
        dist01 = model.forward(img0, img1)
        all_dist01.append(dist01)
        print('%s: %.3f' % (file, dist01))
        f.writelines('%s: %.6f\n' % (file, dist01))

all_dist01 = np.asarray(all_dist01, dtype=np.float32)
print('mean: {} std: {}'.format(all_dist01.mean(), all_dist01.std()))
f.writelines('mean: {} std: {}'.format(all_dist01.mean(), all_dist01.std()))
f.close()
