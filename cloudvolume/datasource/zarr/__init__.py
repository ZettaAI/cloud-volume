from .image import ZarrImageSource
from .metadata import ZarrMetadata

from ...cacheservice import CacheService
from ...cloudvolume import SharedConfiguration, register_plugin
from ...frontends.zarr import CloudVolumeZarr
from ...paths import strict_extract

def create_zarr(
    cloudpath, mip=0, bounded=True, autocrop=False,
    fill_missing=False, cache=False, compress_cache=None,
    cdn_cache=True, progress=False, info=None, provenance=None,
    compress=None, compress_level=None, non_aligned_writes=False, parallel=1,
    delete_black_uploads=False, background_color=0,
    green_threads=False, use_https=False, **kwargs
  ):
    if info is not None:
      raise ValueError("Zarr arrays currently do not support input info files")
    if provenance is not None:
      raise ValueError("Zarr arrays currently do not provenance files")
    if mip != 0:
      raise ValueError("Zarr arrays currently must have MIP=0")
    if fill_missing is False:
      raise ValueError("Zarr arrays currently do not support disabling `fill_missing` flag")

    config = SharedConfiguration(
      cdn_cache=cdn_cache,
      compress=compress,
      compress_level=None,
      green=green_threads,
      mip=mip,
      parallel=parallel,
      progress=progress,
    )

    cache = CacheService(
      cloudpath=(cache if type(cache) == str else cloudpath),
      enabled=bool(cache),
      config=config,
      compress=compress_cache,
    )

    meta = ZarrMetadata(
      cloudpath, cache=cache
    )

    image = ZarrImageSource(
      config, meta, cache,
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=False,
      fill_missing=bool(fill_missing),
      delete_black_uploads=bool(delete_black_uploads)
    )

    return CloudVolumeZarr(
      meta, cache, config,
      image,
    )

def register():
  register_plugin('zarr', create_zarr)
