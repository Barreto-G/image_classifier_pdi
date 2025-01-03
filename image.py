import os, json, cv2, itertools
import numpy as np
import albumentations as A

class Image:
    def __init__(self, name="", bbox=[], mask=[], category=-1, content=None):
        self.content = content
        self.category_id = category
        self.bbox = bbox
        self.mask = mask
        self.file_name = name


    def _rle_to_mask(self, rle_code: list, mask_shape: tuple):
        """
        Convert RLE (Run Length Encoding) to a binary mask.

        Args:
            rle_counts (list): RLE encoded segmentation.
            mask_size (tuple): Size of the mask.

        Returns:
            numpy.ndarray: Binary mask.
        """
        mask = np.zeros(mask_shape[0] * mask_shape[1], dtype=np.uint8)
        current_pos = 0
        for i, count in enumerate(rle_code):
            if i % 2 == 0:
                current_pos += count
            else:
                mask[current_pos:current_pos + count] = 1
                current_pos += count
        return mask.reshape(mask_shape, order='F')
    
    def _mask_to_rle(self, mask) -> list:
        """
        Convert a binary mask to RLE (Run Length Encoding).

        Args:
            mask (numpy.ndarray): Binary mask.

        Returns:
            list: RLE encoded mask.
        """
        pixels = mask.flatten(order='F')
        rle = [sum(1 for _ in group)
               for pixel, group in itertools.groupby(pixels)]
        rle = [0] + rle if pixels[0] else rle
        return rle

    def __repr__(self):
        return f'file_name:{self.file_name},category:{self.category_id},bbox:{self.bbox},mask:{self.mask},content:{self.content}'

