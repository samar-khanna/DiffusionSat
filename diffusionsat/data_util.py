import torch
import random
import rasterio.warp
from rasterio.crs import CRS
import numpy as np
from pyproj import Geod
from shapely.geometry import shape as shapey
from shapely.wkt import loads as shape_loads
import webdataset as wds

from .satlas_util import TASKS, mercator_to_geo
from .fmow_dataset import CATEGORIES, CODE3, CODE2


class SampleEqually(wds.DataPipeline, wds.compat.FluidInterface):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            for source in sources:
                try:
                    yield next(source)
                except StopIteration:
                    return


def fmow_tokenize_caption(example, tokenizer, md_key='metadata.json', drop_pct=0.1, is_train=True, return_text=False):
    str_incl = lambda x: x if random.random() > drop_pct else ''

    if 'output.cls' in example:
        cls_name = ' '.join(CATEGORIES[example['output.cls']].split('_'))
    else:
        cls_name = ' '.join(example['category.txt'].split('_'))
    md = example[md_key]
    gsd = md['gsd']
    cloud_cover = md['cloud_cover']
    country = CODE3.get(md['country_code'], CODE2.get(md['country_code'], md['country_code']))

    caption = (f"a{str_incl(' fmow')} satellite image"
               f"{str_incl(f' of a {cls_name}')}"
               f"{str_incl(f' in {country}')}")
    # f'taken at a gsd of {gsd} with {cloud_cover} cloud cover'
    if return_text:
        return caption

    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids.squeeze()


def spacenet_tokenize_caption(example, tokenizer, drop_pct=0.1, is_train=True, return_text=False):
    str_incl = lambda x: x if random.random() > drop_pct else ''

    key = example['__key__']
    city = key.split('_')[3].title()  # city name

    md = example['metadata.json']
    item_str = ''

    geod = Geod(ellps="WGS84")
    if 'features' in md and len(md['features']) > 0:
        num_items = len(md['features'])
        if 'Polygon' in md['features'][0]['geometry']['type']:
            item_type = 'buildings'
            item_str = f'{num_items} {item_type}'

            total_area = 0.
            for feat in md['features']:
                geom = feat['geometry']
                if city == 'Rotterdam':
                    geom = rasterio.warp.transform_geom(CRS.from_epsg(32631), CRS.from_epsg(4326), geom)

                shape_obj = shapey(geom)
                area, perim = geod.geometry_area_perimeter(shape_obj)
                total_area += abs(area)

            if total_area > 0:
                item_str = item_str + f' covering an area of {round(total_area, 3)} squared meters'

        else:
            total_length = 0
            for feat in md['features']:
                geom = feat['geometry']
                if city == 'Rotterdam':
                    geom = rasterio.warp.transform_geom(CRS.from_epsg(32631), CRS.from_epsg(4326), geom)

                shape_obj = shapey(geom)
                total_length += geod.geometry_length(shape_obj)

            if total_length > 0:
                item_str = f'roads of length {round(total_length, 3)} meters'
            else:
                item_str = 'no roads'

    caption = (f"a{str_incl(' spacenet')} satellite image"
               f"{str_incl(f' of {item_str}') if len(item_str) > 0 else ''}"
               f"{str_incl(f' in {city}')}")
    if return_text:
        return caption

    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt'
    )
    return inputs.input_ids.squeeze()


def satlas_tokenize_caption(example, tokenizer, drop_pct=0.5, is_train=True, return_text=False):
    str_incl = lambda x: x if random.random() > drop_pct else ''

    task_name = example['info.json']['task_name']
    task_info = TASKS[task_name]
    task_type = task_info['type']

    image_names = example['info.json']['image_names']
    targets = example['targets.pyd'][0]

    h, w = example['rgb.npy'].shape[1:3]
    if task_type == 'detect' or task_type == 'instance':
        labels = targets['labels']
        boxes = targets['boxes']

        if random.random() > 0.33:
            positions = {'top left': [], 'top right': [], 'bottom right': [], 'bottom left': []}
            for label, (min_x, min_y, max_x, max_y) in zip(labels, boxes):
                center_x = 0.5 * (min_x + max_x)
                center_y = 0.5 * (min_y + max_y)
                if center_y < h/2 and center_x < h/2:
                    positions['top left'].append(label)
                elif center_y < h/2 and center_x >= h/2:
                    positions['top right'].append(label)
                elif center_y >= h/2 and center_x >= h/2:
                    positions['bottom right'].append(label)
                elif center_y >= h/2 and center_x < h/2:
                    positions['bottom left'].append(label)

            item_str = ''
            for position, label_list in positions.items():
                if len(label_list) > 0:
                    for cls_idx, count in zip(*np.unique(label_list, return_counts=True)):
                        cls_name = ' '.join(task_info['categories'][cls_idx].split('_'))
                        item_str = item_str + f", {count} {cls_name}{'s' if count > 1 else ''}"
                    item_str = item_str + f" in the {position}"

        else:
            item_str = ''
            for cls_idx, count in zip(*np.unique(labels, return_counts=True)):
                cls_name = ' '.join(task_info['categories'][cls_idx].split('_'))
                item_str = item_str + f", {count} {cls_name}{'s' if count > 1 else ''}"

    elif task_type == 'segment':
        mask = targets['im']

        item_str = ''
        for cls_idx, count in zip(*np.unique(mask, return_counts=True)):
            if cls_idx == 0: continue
            cls_name = ' '.join(task_info['categories'][cls_idx].split('_'))
            item_str = item_str + f", {round(100 * count / mask.size)}% {cls_name}"
    elif task_type == 'bin_segment':
        masks = targets['im']

        item_str = ''
        for cls_idx, mask in enumerate(masks):
            cls_name = ' '.join(task_info['categories'][cls_idx].split('_'))
            count = (mask > 0).sum()
            if count > 0:
                cls_name = 'road' if cls_idx == 0 else cls_name
                item_str = item_str + f", {round(100 * count / mask.size, 2)}% {cls_name}"
    elif task_type == 'regress':
        mask = targets['im']

        item_str = ''
        for val, count in zip(*np.unique(mask, return_counts=True)):
            if val == 0: continue
            val = round(val / 255 * 100, 4)
            item_str = item_str + f", {round(100 * count / mask.size, 2)}% area has {val}% {' '.join(task_name.split('_'))}"
    elif task_type == 'classification':
        labels = targets['label']
        task_name_suffix = ' '.join(task_name.split('_')[:-1])
        if task_name == 'park_sport':
            task_name_suffix = 'recreational facility'
        elif task_name == 'park_type':
            task_name_suffix = ''

        item_str = ''
        for cls_idx in labels:
            cls_name = ' '.join(task_info['categories'][cls_idx].split('_'))
            item_str = item_str + f", a {cls_name} {task_name_suffix}"

    else:
        raise NotImplementedError

    item_str = item_str[2:]

    caption = (f"a{str_incl(' satlas')} satellite image of {item_str}")
    # f'taken at a gsd of {gsd} with {cloud_cover} cloud cover'
    if return_text:
        return caption

    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids.squeeze()


def texas_tokenize_caption(example, tokenizer, drop_pct=0.5, is_train=True, return_text=False):
    str_incl = lambda x: x if random.random() > drop_pct else ''
    metadata = example['tif.metadata.json']
    year_built = metadata['eff.year.built']
    acres = metadata['acres']

    caption = (f"a{str_incl(' satlas')} satellite image"
               f"{str_incl(f' of houses ')}"
               f"{str_incl(f' built in {int(year_built)}')}"
               f"{str_incl(f' covering {acres} acres')}")
    if return_text:
        return caption

    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids.squeeze()

def xbd_tokenize_caption(example, tokenizer, md_key='post-metadata.json', drop_pct=0.1, is_train=True, return_text=False):
    str_incl = lambda x: x if random.random() > drop_pct else ''
    metadata = example[md_key]

    disaster_type = metadata["metadata"]["disaster_type"]
    disaster_info = metadata["metadata"]["disaster"]

    before_after_str = "before" if 'pre' in md_key else "after"

    caption = (f"a{str_incl(' fmow')} satellite image"
               f"{str_incl(f' {before_after_str} being affected by a {disaster_type} natural disaster')}")
               # f"{str_incl(f' in {country}')}")
    # f'taken at a gsd of {gsd} with {cloud_cover} cloud cover'
    if return_text:
        return caption

    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids.squeeze()


def is_invalid_lon_lat(lon, lat):
    return np.isnan(lon) or np.isnan(lat) or \
        (lon in [float('inf'), float('-inf')]) or (lat in [float('inf'), float('-inf')]) or \
        lon < -180 or lon > 180 or lat < -90 or lat > 90


def fmow_numerical_metadata(example, meta_df, target_resolution, num_metadata, rgb_key='input.npy',
                            md_key='metadata.json', base_year=1980, base_lon=180, base_lat=90):
    md = example[md_key]
    h, w, c = example[rgb_key].shape
    assert c == 3, 'Shape error'
    orig_res = min(h, w)

    target_res = target_resolution
    scale = orig_res / target_res
    gsd = md['gsd'] * scale

    cloud_cover = md['cloud_cover'] / 100.

    timestamp = md['timestamp']
    year = int(timestamp[:4]) - base_year
    month = int(timestamp[5:7])
    day = int(timestamp[8:10])
    hour = int(timestamp[11:13])

    # name_items = example['__key__'].split('-')[-1].replace('_rgb', '').split('_')
    name_items = example[md_key]['img_filename'].replace('_rgb', '').replace('.jpg', '').replace('_ms.tif', '').split('_')
    category = CATEGORIES[example['output.cls']] if 'output.cls' in example else example['category.txt']  # eg: recreational_facility
    location_id = int(name_items[-2])  # eg: 890
    image_id = int(name_items[-1])  # eg: 4

    polygon = meta_df[
        (meta_df['category'] == category) &
        (meta_df['location_id'] == location_id) &
        (meta_df['image_id'] == image_id)
    ]['polygon']
    assert len(polygon) == 1, f"{category}, {location_id}, {image_id} is not found in csv"
    poly = shape_loads(polygon.iloc[0])
    lon, lat = poly.centroid.x, poly.centroid.y
    assert not is_invalid_lon_lat(lon, lat)

    return torch.tensor([lon + base_lon, lat + base_lat, gsd, cloud_cover, year, month, day])


def spacenet_numerical_metadata(example, target_resolution, num_metadata, base_year=1980, base_lon=180, base_lat=90):
    md = example['metadata.json']
    key = example['__key__']
    city = key.split('_')[3].title()  # city name

    max_lon, min_lon = float('-inf'), float('inf')
    max_lat, min_lat = float('-inf'), float('inf')
    if 'features' in md:
        for feat in md['features']:
            geom = feat['geometry']
            if city == 'Rotterdam':
                geom = rasterio.warp.transform_geom(CRS.from_epsg(32631), CRS.from_epsg(4326), geom)

            shape_obj = shapey(geom)
            lon1, lat1, lon2, lat2 = shape_obj.bounds
            max_lon = max(max_lon, lon2)
            min_lon = min(min_lon, lon1)
            max_lat = max(max_lat, lat2)
            min_lat = min(min_lat, lat1)

    lon, lat = (max_lon + min_lon) / 2, (max_lat + min_lat) / 2
    use_default_coords = np.isnan(lon) or np.isnan(lat) or \
                         (lon in [float('inf'), float('-inf')]) or (lat in [float('inf'), float('-inf')])

    if city in ('Rio', 'Rotterdam'):
        gsd = 0.5
        if city == 'Rio':
            year, month, day = 0, 0, 0
            if use_default_coords:
                lon, lat = -43.196388, -22.908333
        else:
            year, month, day = 2019 - base_year, 8, 31
            if use_default_coords:
                lon, lat = 4.462456, 51.926517
    else:
        gsd = 0.3
        if city == 'Vegas':
            year, month, day = 2015 - base_year, 10, 22
            if use_default_coords:
                lon, lat = -115.176468, 36.188110
        elif city == 'Paris':
            year, month, day = 2016 - base_year, 2, 29
            if use_default_coords:
                lon, lat = 2.349014, 48.864716
        elif city == 'Shanghai':
            year, month, day = 2015 - base_year, 6, 6
            if use_default_coords:
                lon, lat = 121.469170, 31.224361
        elif city == 'Khartoum':
            year, month, day = 2015 - base_year, 4, 13
            if use_default_coords:
                lon, lat = 32.522854, 15.508457
        elif city == 'Moscow':
            year, month, day = 2018 - base_year, 2, 13
            if use_default_coords:
                lon, lat = 37.618423, 55.751244
        elif city == 'Mumbai':
            year, month, day = 2018 - base_year, 1, 6
            if use_default_coords:
                lon, lat = 72.877426, 19.076090
        else:
            year, month, day = 0, 0, 0
    shape = example['rgb.npy'].shape
    h, w = shape[1:] if shape[0] == 3 else shape[:2]

    orig_res = min(h, w)
    target_res = target_resolution
    scale = orig_res / target_res
    gsd = gsd * scale

    assert not is_invalid_lon_lat(lon, lat), f"{key} not valid lat lon"

    return torch.tensor([lon + base_lon, lat + base_lat, gsd, 0, year, month, day])


def satlas_numerical_metadata(example, img_name, target_resolution, num_metadata, base_gsd=1., base_year=1980, base_lon=180, base_lat=90):
    key = example["__key__"]
    h, w = example['rgb.npy'].shape[1:3]

    orig_res = min(h, w)
    target_res = target_resolution
    scale = orig_res / target_res
    resize_scale = 1 / max(example['targets.pyd'][0].get('scale_factor', 1.), 1e-2)
    gsd = base_gsd * scale * resize_scale

    date = img_name.split('_')[-1]
    year, month, day = date.split('-')

    tile_a, tile_b = key.split('_')[1:3]
    lon, lat = mercator_to_geo((int(tile_a), int(tile_b)), zoom=17 if base_gsd == 1. else 13, pixels=target_res)
    assert not is_invalid_lon_lat(lon, lat)

    return torch.tensor([lon + base_lon, lat + base_lat, gsd, 0, int(year) - base_year, int(month), int(day)])


def texas_numerical_metadata(example, img_key, target_resolution, num_metadata, base_gsd=1., base_year=1980, base_lon=180, base_lat=90):
    metadata = example['tif.metadata.json']
    lon = metadata['parcel.lon']
    lat = metadata['parcel.lat']
    cloud_cover = 0

    h, w = example[img_key].shape[:2]
    orig_res = min(h, w)
    target_res = target_resolution
    scale = orig_res / target_res
    base_gsd = 10.0 if 'sentinel' in img_key else 1.0
    gsd = base_gsd * scale

    if '2016' in img_key:
        year, month, day = 2016 - base_year, 9, 11
    else:
        year, month, day = 2018 - base_year, 11, 20  # roughly

    return torch.tensor([lon + base_lon, lat + base_lat, gsd, cloud_cover, year, month, day])


def xbd_numerical_metadata(example, target_resolution, num_metadata, rgb_key='post-input.npy',
                            md_key='post-metadata.json', base_year=1980, base_lon=180, base_lat=90):
    md = example[md_key]
    h, w, c = example[rgb_key].shape
    assert c == 3, 'Shape error'
    orig_res = min(h, w)

    target_res = target_resolution
    scale = orig_res / target_res
    gsd = md['metadata']['gsd'] * scale

    cloud_cover = 0.

    timestamp = md['metadata']['capture_date']
    year = int(timestamp[:4]) - base_year
    month = int(timestamp[5:7])
    day = int(timestamp[8:10])
    hour = int(timestamp[11:13])

    buildings = md['features']['lng_lat']
    if len(buildings) == 0:
        other_md_key = 'pre-metadata.json' if md_key == 'post-metadata.json' else 'post-metadata.json'
        buildings = example[other_md_key]['features']['lng_lat']

    lon, lat = 0, 0
    if len(buildings) > 0:
        polygon = buildings[0]['wkt']
        poly = shape_loads(polygon)
        lon, lat = poly.centroid.x, poly.centroid.y
        assert not is_invalid_lon_lat(lon, lat)
        lon = lon + base_lon
        lat = lat + base_lat

    return torch.tensor([lon, lat, gsd, cloud_cover, year, month, day])


def metadata_normalize(metadata, base_lon=180, base_lat=90, base_year=1980, max_gsd=1., scale=1000):
    lon, lat, gsd, cloud_cover, year, month, day = metadata
    lon = lon / (180 + base_lon) * scale
    lat = lat / (90 + base_lat) * scale
    gsd = gsd / max_gsd * scale
    cloud_cover = cloud_cover * scale
    year = year / (2100 - base_year) * scale
    month = month / 12 * scale
    day = day / 31 * scale
    return torch.tensor([lon, lat, gsd, cloud_cover, year, month, day])


def metadata_unnormalize(norm_metadata, base_lon=180, base_lat=90, base_year=1980, max_gsd=1., scale=1000,
                         is_print=False):
    lon, lat, gsd, cloud_cover, year, month, day = norm_metadata
    lon = lon / scale * (180 + base_lon) - (base_lon if is_print else 0)
    lat = lat / scale * (90 + base_lat) - (base_lat if is_print else 0)
    gsd = gsd / scale * max_gsd
    cloud_cover = cloud_cover / scale
    year = year / scale * (2100 - base_year) + (base_year if is_print else 0)
    month = month / scale * 12
    day = day / scale * 31
    return torch.tensor([lon, lat, gsd, cloud_cover, year, month, day])


def combine_text_and_metadata(text_caption, metadata, tokenizer, return_text=False):

    lon, lat = metadata[0].item(), metadata[1].item()
    gsd = metadata[2].item()
    year, month, day = metadata[4].item(), metadata[5].item(), metadata[6].item()
    text_caption += f' at a resolution of {round(gsd, 5)}. The longitude, latitude is {lon}, {lat}. '
    if year != 0. and month != 0. and day != 0.:
        text_caption += f'The date is {year}, {month}, {day}'
    if return_text:
        return text_caption

    inputs = tokenizer(
        text_caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids.squeeze()

def fmow_temporal_images(example, img_transform, num_frames=3, is_random=True, stack_tensor=True, channel_first=False):
    image_keys = sorted([k for k in example if k.endswith('.npy')])
    metadata_keys = sorted([k for k in example if k.endswith('.json')])
    if len(image_keys) < num_frames:
        while len(image_keys) < num_frames:
            image_keys.append('input-0.npy')
            metadata_keys.append('metadata-0.json')
    else:
        img_md = random.sample(list(zip(image_keys, metadata_keys)), k=num_frames) \
            if is_random else list(zip(image_keys, metadata_keys))[:num_frames]
        image_keys = [img for img, md in img_md]
        metadata_keys = [md for img, md in img_md]

    img = [img_transform(example[k]) for k in image_keys]
    if stack_tensor:
        img = torch.stack(img)  # (T, C, H, W)
        if channel_first:
            img = img.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

    return img, metadata_keys


def percentile_normalization(
    img,
    lower: float = 2,
    upper: float = 98,
    axis = None,
):
    """
    Borrowed from: https://torchgeo.readthedocs.io/en/stable/_modules/torchgeo/datasets/utils.html
    Applies percentile normalization to an input image.

    Specifically, this will rescale the values in the input such that values <= the
    lower percentile value will be 0 and values >= the upper percentile value will be 1.
    Using the 2nd and 98th percentile usually results in good visualizations.

    Args:
        img: image to normalize
        lower: lower percentile in range [0,100]
        upper: upper percentile in range [0,100]
        axis: Axis or axes along which the percentiles are computed. The default
            is to compute the percentile(s) along a flattened version of the array.

    Returns
        normalized version of ``img``
    """
    assert lower < upper
    lower_percentile = np.percentile(img, lower, axis=axis)
    upper_percentile = np.percentile(img, upper, axis=axis)
    img_normalized = np.clip(
        (img - lower_percentile) / (upper_percentile - lower_percentile + 1e-5), 0, 1
    )
    return img_normalized


class SentinelNormalize:
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean=None, std=None, channel_specific=True):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std

        self.mean = np.array(mean)
        self.std = np.array(std)
        if not channel_specific:
            self.mean = self.mean.mean()
            self.std = self.std.mean()

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelDropBands:
    """
    Must call after conversion to Tensor
    """
    def __init__(self, dropped_bands=[],):
        self.dropped_bands = dropped_bands

    def __call__(self, x, *args, **kwargs):
        keep_idxs = [i for i in range(x.shape[0]) if i not in self.dropped_bands]
        x = x[keep_idxs, :, :]
        return x


class SentinelFlipBGR:
    """
    Must call after conversion to Tensor
    """
    def __init__(self):
        pass

    def __call__(self, x, *args, **kwargs):
        x[1:4, :, :] = x[[3,2,1], :, :]
        return x


class IdentityTransform:
    def __init__(self):
        pass

    def __ceil__(self, x, *args, **kwargs):
        return x
