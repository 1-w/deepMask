import os
import re
import fileinput
import time
import subprocess
import numpy as np
from sklearn.utils import class_weight
import matplotlib.cm as cm
from scipy import ndimage
import nibabel as nib
from nibabel import load as load_nii
import torch
from torch.autograd import Variable
from skimage import transform as skt


def deepMask(args, model, id, t1w_np, t2w_np, t1w_fname, t2w_fname, nifti=True):
    dst = args.outdir
    case_id = id

    model.eval()

    # for sample in loader:
    start_time = time.time()
    t1w_np = skt.resize(t1w_np, args.resize, mode='constant', preserve_range=1)
    t2w_np = skt.resize(t2w_np, args.resize, mode='constant', preserve_range=1)
    data = torch.from_numpy(np.stack((t1w_np, t2w_np), axis=0))

    # load original input with header and affine
    _, header, affine, out_shape = get_nii_hdr_affine(t1w_fname)

    shape = data.size()
    # convert names to batch tensor
    if args.cuda:
        data.pin_memory()
        data = data.cuda().float()
    else:
        data = data.float()
    data = Variable(data, volatile=True)
    output = model(data)
    output = torch.argmax(output, dim=1)
    output = output.view(shape[2:])
    output = output.cpu()
    output = output.data.numpy()

    print("save {}".format(case_id))
    if not os.path.exists(os.path.join(dst)):
        os.makedirs(os.path.join(dst), exist_ok=True)

    if nifti:
        affine = header.get_qform()
        output = skt.resize(output, output_shape=out_shape, order=1, mode='wrap', preserve_range=1, anti_aliasing=True)
        nii_out = nib.Nifti1Image(output, affine, header)
        nii_out.to_filename(os.path.join(dst, case_id+"_vnet_maskpred.nii.gz"))

    elapsed_time = time.time() - start_time
    print("======")
    print("=> inference time: {} seconds".format(round(elapsed_time,2)))
    print("======")

    config = '/app/dense3dCrf/config_densecrf.txt'

    start_time = time.time()
    denseCRF(case_id, t1w_fname, t2w_fname, out_shape, config, dst, os.path.join(dst, case_id+"_vnet_maskpred.nii.gz"))
    elapsed_time = time.time() - start_time
    print("======")
    print("=> dense 3D-CRF inference time: {} seconds".format(round(elapsed_time,2)))
    print("======")

    fname = os.path.join(dst, case_id + '_denseCrf3dSegmMap.nii.gz')
    seg_map = load_nii(fname).get_fdata()
    return seg_map


def denseCRF(id, t1, t2, input_shape, config, out_dir, pred_labels):
    X, Y, Z = input_shape
    config_tmp = "/tmp/" + id + "_config_densecrf.txt"
    print(out_dir)
    subprocess.call(["cp", "-f", config, config_tmp])
    # find and replace placeholder with actual filenames
    find_str = [
                "<ID_PLACEHOLDER>", "<T1_FILE_PLACEHOLDER>", "<FLAIR_FILE_PLACEHOLDER>",
                "<OUTDIR_PLACEHOLDER>", "<PRED_LABELS_PLACEHOLDER>",
                "<X_PLACEHOLDER>", "<Y_PLACEHOLDER>", "<Z_PLACEHOLDER>"
                ]
    replace_str = [
                    str(id), str(t1), str(t2),
                    str(out_dir), str(pred_labels),
                    str(X), str(Y), str(Z)
                    ]

    for fs, rs in zip(find_str, replace_str):
        find_replace_re(config_tmp, fs, rs)
    subprocess.call(["/app/dense3dCrf/dense3DCrfInferenceOnNiis", "-c", config_tmp])


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def find_replace_re(config_tmp, find_str, replace_str):
    with fileinput.FileInput(config_tmp, inplace=True, backup='.bak') as file:
        for line in file:
            print(re.sub(find_str, str(replace_str), line.rstrip(), flags=re.MULTILINE), end='\n')





def compute_weights(labels, binary=False):
	if binary:
		labels = (labels > 0).astype(np.int_)

	weights = class_weight.compute_class_weight('balanced', np.unique(labels.flatten()), labels.flatten())
	return weights


def dice_gross(image, label, empty_score=1.0):

    image = (image > 0).astype(np.int_)
    label = (label > 0).astype(np.int_)

    image = np.asarray(image).astype(np.bool)
    label = np.asarray(label).astype(np.bool)

    if image.shape != label.shape:
        raise ValueError("Shape mismatch: image {0} and label {1} must have the same shape.".format(image.shape, label.shape))

    im_sum = image.sum() + label.sum()
    if im_sum == 0:
        return empty_score

    # compute Dice coefficient
    intersection = np.logical_and(image, label)

    return 2. * intersection.sum() / im_sum


def get_nii_hdr_affine(t1w_fname):
    nifti = load_nii(t1w_fname)
    shape = nifti.get_fdata().shape
    header = load_nii(t1w_fname).header
    affine = header.get_qform()
    return nifti, header, affine, shape


def plotslice(i, image, label, ax, fig, x, y, z, alfa=0.5, binarize=False, mask=False):

    if binarize:
        label = (label > 0).astype(np.int_)

    if mask:
        label = (label > 0).astype(np.int_)
        t1w_bin = (image > 0).astype(np.int_)
        mask = t1w_bin + label
        label = t1w_bin

    ax = plt.subplot(1, 3, 1)
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(ndimage.rotate(image[x,:,:],90), cmap=cm.gray)
    plt.imshow(ndimage.rotate(label[x,:,:],90), cmap=cm.viridis, alpha=alfa)

    ax = plt.subplot(1, 3, 2)
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(ndimage.rotate(image[:,y,:],90), cmap=cm.gray)
    plt.imshow(ndimage.rotate(label[:,y,:],90), cmap=cm.viridis, alpha=alfa)

    ax = plt.subplot(1, 3, 3)
    ax.set_title('Sample #{}'.format(i), )
    ax.axis('off')
    plt.imshow(ndimage.rotate(image[:,:,z],90), cmap=cm.gray)
    plt.imshow(ndimage.rotate(label[:,:,z],90), cmap=cm.viridis, alpha=alfa)