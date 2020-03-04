from __future__ import print_function

import numpy as np
import zarr
from zarr.core import Array

from ... import exceptions
from ...lib import (Bbox, Vec, colorize, generate_random_string, jsonify, max2,
                    min2, mkdir, red)
from ...storage import ThreadedStorage
from ...volumecutout import VolumeCutout
from .. import ImageSourceInterface, autocropfn, readonlyguard

class ZarrImageSource(ImageSourceInterface):
  def __init__(
    self, config, meta, cache,
    autocrop=False, bounded=True,
    non_aligned_writes=False,
    fill_missing=False,
    delete_black_uploads=False,
    background_color=0,
    readonly=True,
  ):
    self.config = config
    self.meta = meta
    self.cache = cache

    self.autocrop = bool(autocrop)
    self.bounded = bool(bounded)
    self.readonly = bool(readonly)
    self.background_color = background_color

  def check_bounded(self, bbox):
    if self.bounded and not self.meta.bounds().contains_bbox(bbox):
      raise exceptions.OutOfBoundsError("""
        Requested cutout not contained within dataset bounds.

        Cloudpath: {}
        Requested: {}
        Bounds: {}
        Resolution: {}

        Set bounded=False to disable this warning.
      """.format(
          self.meta.cloudpath,
          bbox, self.meta.bounds(),
          self.meta.resolution()
        )
      )

  def download(self, bbox, parallel=1):
    bounds = bbox.clone()
    if self.autocrop:
      bbox = Bbox.intersection(bbox, self.meta.bounds())
    else:
      self.check_bounded(bbox)

    with ThreadedStorage(self.meta.cloudpath) as stor:
      src = Array(store=stor, read_only=True)
      src._fill_value = self.background_color
      cutout = src.get_basic_selection(bbox.to_slices())

    if len(cutout.shape) == 3:
      cutout = cutout.reshape(tuple(list(cutout.shape) + [ 1 ]))

    if self.bounded or self.autocrop or bounds == bbox:
      return VolumeCutout.from_volume(self.meta, 0, cutout, bbox)

    # This section below covers the case where the requested volume is bigger
    # than the dataset volume and the bounds guards have been switched 
    # off. This is useful for Marching Cubes where a 1px excess boundary
    # is needed.
    shape = list(bbox.size3()) + [ cutout.shape[3] ]
    renderbuffer = np.zeros(shape=shape, dtype=self.meta.dtype, order=self.meta.order)
    shade(renderbuffer, bbox, cutout, bounds)
    return VolumeCutout.from_volume(self.meta, 0, renderbuffer, bbox)

  @readonlyguard
  def upload(self, image, offset, mip):
    raise NotImplementedError()

  def exists(self, bbox, mip=None):
    raise NotImplementedError()

  @readonlyguard
  def delete(self, bbox, mip=None):
    raise NotImplementedError()

  def transfer_to(self, cloudpath, bbox, mip, block_size=None, compress=True):
    raise NotImplementedError()


def shade(dest_img, dest_bbox, src_img, src_bbox):
  """
  Shade dest_img at coordinates dest_bbox using the
  image contained in src_img at coordinates src_bbox.

  The buffer will only be painted in the overlapping
  region of the content.

  Returns: void
  """
  if not Bbox.intersects(dest_bbox, src_bbox):
    return

  spt = max2(src_bbox.minpt, dest_bbox.minpt)
  ept = min2(src_bbox.maxpt, dest_bbox.maxpt)
  dbox = Bbox(spt, ept) - dest_bbox.minpt

  ZERO3 = Vec(0, 0, 0)
  istart = max2(spt - src_bbox.minpt, ZERO3)
  iend = min2(ept - src_bbox.maxpt, ZERO3) + src_img.shape[:3]
  sbox = Bbox(istart, iend)

  while src_img.ndim < 4:
    src_img = src_img[..., np.newaxis]

  dest_img[ dbox.to_slices() ] = src_img[ sbox.to_slices() ]
