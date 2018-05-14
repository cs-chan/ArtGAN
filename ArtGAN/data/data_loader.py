import numpy as np
from neon.data.aeon_shim import AeonDataLoader
from neon.data.dataloader_transformers import OneHot, TypeCast
from neon.util.persist import get_data_cache_or_nothing


def common_config(manifest_file, manifest_root, batch_size, subset_pct, h=96, w=96, scale=[1., 1.]):
    cache_root = get_data_cache_or_nothing('stl10-cache/')
    return {
               'manifest_filename': manifest_file,
               'manifest_root': manifest_root,
               'minibatch_size': batch_size,
               'subset_fraction': float(subset_pct/100.0),
               'macrobatch_size': 5000,
               'type': 'image,label',
               'cache_directory': cache_root,
               'image': {'height': h,
                         'width': w,
                         'scale': scale},
               'label': {'binary': False}
            }


def wrap_dataloader(dl, ncls=10):
    dl = OneHot(dl, index=1, nclasses=ncls)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


def train_loader(manifest_file, manifest_root, backend_obj, subset_pct=100, random_seed=0, h=96, w=96, scale=[1., 1.], binary=False):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct, h, w, scale)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed
    aeon_config['image']['center'] = False
    aeon_config['image']['flip_enable'] = True
    aeon_config['label']['binary'] = binary
    # return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj))
    return AeonDataLoader(aeon_config, backend_obj)


def validation_loader(manifest_file, manifest_root, backend_obj, subset_pct=100, h=96, w=96, scale=[1., 1.], ncls=10):
    aeon_config = common_config(manifest_file, manifest_root, backend_obj.bsz, subset_pct, h, w, scale)
    aeon_config['image']['center'] = True
    return wrap_dataloader(AeonDataLoader(aeon_config, backend_obj), ncls=ncls)
