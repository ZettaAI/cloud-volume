import os
import posixpath

from zarr.core import Array

from cloudvolume import Bbox, CloudVolume, Vec
from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata
from cloudvolume.storage import SimpleStorage

from ...paths import strict_extract

# Zarr Metadata will try to list all files in the remote directory to count
# "initialized" chunks - need to prevent that.
setattr(Array, 'nchunks_initialized', property(lambda self: self.nchunks))

class ZarrMetadata(object):
  """
  The ZarrMetadataService provides methods for fetching and
  accessing information about the data type & compression,
  bounding box, and resolution of a given dataset
  stored in zarr array format.
  """
  def __init__(
    self, cloudpath, cache=None
  ):
    self.path = strict_extract(cloudpath)
    self.cache = cache
    if self.cache:
      self.cache.meta = self

    self.metadata, self.user_attribs = self.refresh_info()

    # Currently only support 3 dimensional, single channel data
    assert len(self.metadata.shape) == 3

    if self.cache and self.cache.enabled:
      self.cache.check_info_validity()

  def refresh_info(self):
    """
    Refresh the current info file from the cache (if enabled) 
    or primary storage (e.g. the cloud) if not cached.

    Raises:
      cloudvolume.exceptions.InfoUnavailableError when the info file 
        is unable to be retrieved.

    See also: fetch_info

    Returns: dict
    """
    if self.cache and self.cache.enabled:
      metadata = self.cache.get_json('metadata')
      user_attribs = self.cache.get_json('user_attribs')
      if metadata:
        self.metadata = metadata
      if user_attribs:
        self.user_attribs = user_attribs

      return self.metadata, self.user_attribs

    self.metadata, self.user_attribs = self.fetch_info()

    if self.cache:
      self.cache.maybe_cache_info()
    return self.metadata, self.user_attribs

  def fetch_info(self):
    with SimpleStorage(self.cloudpath) as stor:
      arr = Array(store=stor, read_only=True)
      metadata = arr.info.obj
      user_attribs = arr.attrs

    return metadata, user_attribs

  @property
  def dataset(self):
    return self.path.dataset
  
  @property
  def layer(self):
    return self.path.layer

  def join(self, *paths):
    if self.path.protocol == 'file':
      return os.path.join(*paths)
    else:
      return posixpath.join(*paths)

  @property
  def basepath(self):
    return self.path.basepath

  @property 
  def layerpath(self):
    return self.join(self.basepath, self.layer)

  @property
  def base_cloudpath(self):
    return self.path.protocol + "://" + self.basepath

  @property 
  def cloudpath(self):
    return self.join(self.base_cloudpath, self.layer)

  def shape(self):
    return Vec(*self.metadata.shape)

  @property
  def dtype(self):
    """e.g. np.uint8"""
    return self.metadata.dtype

  @property
  def data_type(self):
    """e.g. 'uint8'"""
    return str(self.metadata.dtype)

  def encoding(self):
    return self.metadata.encoding

  @property
  def layer_type(self):
    return 'image'

  @property
  def num_channels(self):
    return 1

  @property
  def order(self):
    return self.metadata.order

  def voxel_offset(self):
    """Vec(x,y,z) start of the dataset in voxels"""
    if self.user_attribs:
      return Vec(*self.user_attribs.get('offset', (0, 0, 0)))
    else:
      return Vec(0, 0, 0)

  def resolution(self, mip=None):
    """Vec(x,y,z) dimensions of each voxel in nanometers"""
    if self.user_attribs:
      return Vec(*self.user_attribs.get('resolution', (1, 1, 1)))
    else:
      return Vec(1, 1, 1)

  def chunk_size(self):
    """Underlying chunk size dimensions in voxels. Synonym for underlying."""
    return Vec(*self.metadata.chunks)

  def bounds(self):
    """Returns a 3D spatial bounding box for the dataset with dimensions in voxels."""
    offset = self.voxel_offset()
    shape = self.shape()
    return Bbox(offset, offset + shape)

  def bbox(self):
    bounds = self.bounds()
    minpt = list(bounds.minpt) + [ 0 ]
    maxpt = list(bounds.maxpt) + [ self.num_channels ]
    return Bbox(minpt, maxpt)
