#!/usr/bin/env python3

import os
import sys
import torch
from mo_dots import Data
from utils.data import *
from utils.deepmask import *
from utils.image_processing import noelImageProcessor
import vnet

# configuration
args = Data()

args.outdir = os.path.join(sys.argv[4], str(sys.argv[1]))
args.tmpdir = os.path.join(args.outdir, "tmp")
args.seed = 666

cwd = os.path.dirname(__file__)
# trained weights based on manually corrected masks from
# 153 patients with cortical malformations
args.inference = os.path.join(cwd, "weights", "vnet_masker_model_best.pth.tar")

# resize all input images to this resolution matching training data
args.resize = (160, 160, 160)

args.use_gpu = True
args.cuda = torch.cuda.is_available() and args.use_gpu

torch.manual_seed(args.seed)
args.device_ids = list(range(torch.cuda.device_count()))

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("build vnet, using GPU")
else:
    print("build vnet, using CPU")

model = vnet.build_model(args)

template = os.path.join(cwd, "template", "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz")

args.t1 = os.path.join(args.outdir, sys.argv[2])
args.t2 = os.path.join(args.outdir, sys.argv[3])
noelImageProcessor(
    id=sys.argv[1],
    t1=args.t1,
    t2=args.t2,
    output_suffix="_brain_final.nii.gz",
    output_dir=args.outdir,
    template=template,
    usen3=True,
    args=args,
    model=model,
    preprocess=True,
).pipeline()
