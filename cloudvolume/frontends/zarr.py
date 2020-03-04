from __future__ import print_function

import itertools
import json
import multiprocessing as mp
import os
import socket
import sys
import uuid

import fastremap
import gevent.socket
import numpy as np
from six import string_types
from six.moves import range
from tqdm import tqdm

from .. import exceptions, lib, sharedmemory
from ..cacheservice import CacheService
from ..datasource import autocropfn
from ..datasource.zarr import ZarrMetadata
from ..lib import Bbox, Vec, colorize, jsonify, mkdir, red
from ..paths import strict_extract
from ..provenance import DataLayerProvenance
from ..storage import SimpleStorage, Storage, reset_connection_pools
from ..volumecutout import VolumeCutout


def warn(text):
  print(colorize('yellow', text))

class CloudVolumeZarr(object):
  def __init__(self,
    meta, cache, config,
    image=None,
  ):
    self.config = config
    self.cache = cache
    self.meta = meta

    self.image = image

    self.green_threads = self.config.green # display warning message

    # needs to be set after info is defined since
    # its setter is based off of scales
    self.pid = os.getpid()

  @property
  def autocrop(self):
    return self.image.autocrop

  @autocrop.setter
  def autocrop(self, val):
    self.image.autocrop = val

  @property
  def background_color(self):
    return self.image.background_color

  @background_color.setter
  def background_color(self, val):
    self.image.background_color = val

  @property 
  def bounded(self):
    return self.image.bounded

  @bounded.setter 
  def bounded(self, val):
    self.image.bounded = val

  @property
  def fill_missing(self):
    return self.image.fill_missing

  @fill_missing.setter
  def fill_missing(self, val):
    self.image.fill_missing = val

  @property
  def green_threads(self):
    return self.config.green

  @green_threads.setter 
  def green_threads(self, val):
    if val and socket.socket is not gevent.socket.socket:
      warn("""
      WARNING: green_threads is set but this process is
      not monkey patched. This will cause severely degraded
      performance.
      
      CloudVolume uses gevent for cooperative (green)
      threading but it requires patching the Python standard
      library to perform asynchronous IO. Add this code to
      the top of your program (before any other imports):

        import gevent.monkey
        gevent.monkey.patch_all(threads=False)

      More Information:

      http://www.gevent.org/intro.html#monkey-patching
      """)

    self.config.green = bool(val)

  @property
  def parallel(self):
    return self.config.parallel

  @parallel.setter
  def parallel(self, num_processes):
    if type(num_processes) == bool:
      num_processes = mp.cpu_count() if num_processes == True else 1
    elif num_processes <= 0:
      raise ValueError('Number of processes must be >= 1. Got: ' + str(num_processes))
    else:
      num_processes = int(num_processes)

    self.config.parallel = num_processes

  @property
  def progress(self):
    return self.config.progress

  @progress.setter
  def progress(self, val):
    self.config.progress = bool(val)

  @property
  def info(self):
    return self.meta.metadata.info, self.meta.user_attribs.asdict()

  def __setstate__(self, d):
    """Called when unpickling which is integral to multiprocessing."""
    self.__dict__ = d

    pid = os.getpid()
    if 'pid' in d and d['pid'] != pid:
      # otherwise the pickle might have references to old connections
      reset_connection_pools()
      self.pid = pid

  # @classmethod
  # def create_new_info(cls,
  #   num_channels, layer_type, data_type, encoding,
  #   resolution, voxel_offset, volume_size,
  #   mesh=None, skeletons=None, chunk_size=(64,64,64),
  #   redirect=None, *args, **kwargs
  # ):
  #   """
  #   Create a new neuroglancer Precomputed info file.

  #   Required:
  #     num_channels: (int) 1 for grayscale, 3 for RGB 
  #     layer_type: (str) typically "image" or "segmentation"
  #     data_type: (str) e.g. "uint8", "uint16", "uint32", "float32"
  #     encoding: (str) "raw" for binaries like numpy arrays, "jpeg"
  #     resolution: int (x,y,z), x,y,z voxel dimensions in nanometers
  #     voxel_offset: int (x,y,z), beginning of dataset in positive cartesian space
  #     volume_size: int (x,y,z), extent of dataset in cartesian space from voxel_offset

  #   Optional:
  #     mesh: (str) name of mesh directory, typically "mesh"
  #     skeletons: (str) name of skeletons directory, typically "skeletons"
  #     chunk_size: int (x,y,z), dimensions of each downloadable 3D image chunk in voxels
  #     redirect: If this volume has moved, you can set an automatic redirect
  #       by specifying a cloudpath here.

  #   Returns: dict representing a single mip level that's JSON encodable
  #   """
  #   return ZarrMetadata.create_info(
  #     num_channels, layer_type, data_type, encoding,
  #     resolution, voxel_offset, volume_size,
  #     mesh, skeletons, chunk_size,
  #     *args, **kwargs
  #   )

  def refresh_info(self):
    """Restore the current info from cache or storage."""
    return self.meta.refresh_info()

  @property
  def dataset_name(self):
    return self.meta.dataset

  @property
  def layer(self):
    return self.meta.layer

  @property
  def basepath(self):
    return self.meta.basepath

  @property
  def layerpath(self):
    return self.meta.layerpath

  @property
  def base_cloudpath(self):
    return self.meta.base_cloudpath

  @property
  def cloudpath(self):
    return self.layer_cloudpath

  @property
  def layer_cloudpath(self):
    return self.meta.cloudpath

  @property
  def info_cloudpath(self):
    return self.meta.infopath

  @property
  def cache_path(self):
    return self.cache.path

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def shape(self):
    """Returns Vec(x,y,z,channels) shape of the volume similar to numpy.""" 
    return tuple(self.meta.shape())

  @property
  def volume_size(self):
    """Returns Vec(x,y,z) shape of the volume (i.e. shape - channels).""" 
    return self.meta.volume_size

  @property
  def dtype(self):
    """e.g. 'uint8'"""
    return self.meta.dtype

  @property
  def data_type(self):
    return self.meta.data_type

  @property
  def num_channels(self):
    return self.meta.num_channels

  @property
  def order(self):
    return self.meta.order

  @property
  def voxel_offset(self):
    """Vec(x,y,z) start of the dataset in voxels"""
    return self.meta.voxel_offset()

  @property 
  def resolution(self):
    """Vec(x,y,z) dimensions of each voxel in nanometers"""
    return self.meta.resolution()

  @property
  def downsample_ratio(self):
    """Describes how downsampled the current mip level is as an (x,y,z) factor triple."""
    return 1.0

  @property
  def chunk_size(self):
    """Underlying chunk size dimensions in voxels. Synonym for underlying."""
    return self.meta.chunk_size()

  @property
  def bounds(self):
    """Returns a bounding box for the dataset with dimensions in voxels"""
    return self.meta.bounds()

  def bbox_to_mip(self, bbox, mip, to_mip):
    """Convert bbox or slices from one mip level to another."""
    assert mip == 0 and to_mip == 0
    return Bbox.create(bbox, self.bounds)

  def exists(self, bbox_or_slices):
    """
    Produce a summary of whether all the requested chunks exist.

    bbox_or_slices: accepts either a Bbox or a tuple of slices representing
      the requested volume.
    Returns: { chunk_file_name: boolean, ... }
    """
    return self.image.exists(bbox_or_slices)

  def delete(self, bbox_or_slices):
    """
    Delete the files within the bounding box.

    bbox_or_slices: accepts either a Bbox or a tuple of slices representing
      the requested volume.
    """
    return self.image.delete(bbox_or_slices)

  def __getitem__(self, slices):
    if type(slices) == Bbox:
      slices = slices.to_slices()

    slices = self.meta.bbox().reify_slices(slices, bounded=self.bounded)
    steps = Vec(*[ slc.step for slc in slices ])
    channel_slice = slices.pop()
    requested_bbox = Bbox.from_slices(slices)

    img = self.download(requested_bbox)
    return img[::steps.x, ::steps.y, ::steps.z, channel_slice]

  def download(
      self, bbox, parallel=None
    ):
    """
    Downloads segmentation from the indicated cutout
    region.

    bbox: specifies cutout to fetch
    parallel: what parallel level to use (default self.parallel)

    Returns: img
    """
    bbox = Bbox.create(
      bbox, context=self.bounds,
      bounded=self.bounded,
      autocrop=self.autocrop
    )

    if parallel is None:
      parallel = self.parallel

    img = self.image.download(bbox, parallel=parallel)

    return img

  def unlink_shared_memory(self):
    """Unlink the current shared memory location from the filesystem."""
    return self.image.unlink_shared_memory()

  def download_to_shared_memory(self, slices, location=None):
    """
    Download images to a shared memory array. 

    https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Shared-Memory

    tip: If you want to use slice notation, np.s_[...] will help in a pinch.

    MEMORY LIFECYCLE WARNING: You are responsible for managing the lifecycle of the 
      shared memory. CloudVolume will merely write to it, it will not unlink the 
      memory automatically. To fully clear the shared memory you must unlink the 
      location and close any mmap file handles. You can use `cloudvolume.sharedmemory.unlink(...)`
      to help you unlink the shared memory file or `vol.unlink_shared_memory()` if you do 
      not specify location (meaning the default instance location is used).

    EXPERT MODE WARNING: If you aren't sure you need this function (e.g. to relieve 
      memory pressure or improve performance in some way) you should use the ordinary 
      download method of img = vol[:]. A typical use case is transferring arrays between 
      different processes without making copies. For reference, this  feature was created 
      for downloading a 62 GB array and working with it in Julia.

    Required:
      slices: (Bbox or list of slices) the bounding box the shared array represents. For instance
        if you have a 1024x1024x128 volume and you're uploading only a 512x512x64 corner
        touching the origin, your Bbox would be `Bbox( (0,0,0), (512,512,64) )`.
    Optional:
      location: (str) Defaults to self.shared_memory_id. Shared memory location 
        e.g. 'cloudvolume-shm-RANDOM-STRING' This typically corresponds to a file 
        in `/dev/shm` or `/run/shm/`. It can also be a file if you're using that for mmap. 
    
    Returns: ndarray backed by shared memory
    """

    slices = self.meta.bbox().reify_slices(slices, bounded=self.bounded)
    steps = Vec(*[ slc.step for slc in slices ])
    channel_slice = slices.pop()
    requested_bbox = Bbox.from_slices(slices)

    if self.autocrop:
      requested_bbox = Bbox.intersection(requested_bbox, self.bounds)

    img = self.image.download(
      requested_bbox, parallel=self.parallel,
      location=location, retain=True, use_shared_memory=True
    )
    return img[::steps.x, ::steps.y, ::steps.z, channel_slice]

  def download_to_file(self, path, bbox):
    """
    Download images directly to a file.

    Required:
      slices: (Bbox) the bounding box the shared array represents. For instance
        if you have a 1024x1024x128 volume and you're uploading only a 512x512x64 corner
        touching the origin, your Bbox would be `Bbox( (0,0,0), (512,512,64) )`.
      path: (str)

    Returns: ndarray backed by an mmapped file
    """

    slices = self.meta.bbox().reify_slices(bbox, bounded=self.bounded)
    steps = Vec(*[ slc.step for slc in slices ])
    channel_slice = slices.pop()
    requested_bbox = Bbox.from_slices(slices)

    if self.autocrop:
      requested_bbox = Bbox.intersection(requested_bbox, self.bounds)

    img = self.image.download(
      requested_bbox, parallel=self.parallel,
      location=lib.toabs(path), retain=True, use_file=True
    )
    return img[::steps.x, ::steps.y, ::steps.z, channel_slice]

  def __setitem__(self, slices, img):
    raise NotImplementedError()

  def upload_from_shared_memory(self, location, bbox, order='F', cutout_bbox=None):
    raise NotImplementedError()

  def upload_from_file(self, location, bbox, order='F', cutout_bbox=None):
    raise NotImplementedError()

  def viewer(self, port=1337):
    import cloudvolume.server

    cloudvolume.server.view(self.cloudpath, port=port)

  def to_dask(self, chunks=None, name=None):
    """Return a dask array for this volume.

    Parameters
    ----------
    chunks: tuple of ints or tuples of ints
      Passed to ``da.from_array``, allows setting the chunks on
      initialisation, if the chunking scheme in the stored dataset is not
      optimal for the calculations to follow. Note that the chunking should
      be compatible with an underlying 4d array.
    name: str, optional
      An optional keyname for the array. Defaults to hashing the input

    Returns
    -------
    Dask array
    """
    import dask.array as da
    from dask.base import tokenize

    if chunks is None:
      chunks = tuple(self.chunk_size)
    if name is None:
      name = 'to-dask-' + tokenize(self, chunks)
    return da.from_array(self, chunks, name=name)
