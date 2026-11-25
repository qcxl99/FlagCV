"""Run OpenCV CPU on host snapshots and copy into the supplied tensor."""


def evaluate(name, sources, dst, *options):
    import cv2
    import torch

    arrays = [source.detach().cpu().numpy() for source in sources]
    result = getattr(cv2, name)(*arrays, *options)
    # OpenCV Python may drop the last axis for a single-channel image.
    dst.copy_(torch.from_numpy(result.reshape(tuple(dst.shape))))
    return dst
