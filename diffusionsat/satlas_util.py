###########################################################################
# References:
# https://github.com/allenai/satlas
###########################################################################
import math

def mercator_to_geo(p, zoom=13, pixels=512):
    n = 2**zoom
    x = p[0] / pixels
    y = p[1] / pixels
    x = x * 360.0 / n - 180
    y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    y = y * 180 / math.pi
    return (x, y)  # lon, lat


TASKS = {
    'polyline_bin_segment': {
        'type': 'bin_segment',
        'categories': [
            'airport_runway', 'airport_taxiway', 'raceway', 'road', 'railway', 'river',
        ],
        'colors': [
            [255, 255, 255], # (white) airport_runway
            [192, 192, 192], # (light grey) airport_taxiway
            [160, 82, 45], # (sienna) raceway
            [255, 255, 255], # (white) road
            [144, 238, 144], # (light green) railway
            [0, 0, 255], # (blue) river
        ],
    },
    'bin_segment': {
        'type': 'bin_segment',
        'categories': [
            "aquafarm", "lock", "dam", "solar_farm", "power_plant", "gas_station",
            "park", "parking_garage", "parking_lot", "landfill", "quarry",
            "stadium", "airport", "airport_runway", "airport_taxiway",
            "airport_apron", "airport_hangar", "airstrip", "airport_terminal",
            "ski_resort", "theme_park", "storage_tank", "silo", "track",
            "raceway", "wastewater_plant", "road", "railway", "river",
            "water_park", "pier", "water_tower", "street_lamp", "traffic_signals",
            "power_tower", "power_substation", "building", "bridge",
            "road_motorway", "road_trunk", "road_primary", "road_secondary", "road_tertiary",
            "road_residential", "road_service", "road_track", "road_pedestrian",
        ],
        'colors': [
            [32, 178, 170], # (light sea green) aquafarm
            [0, 255, 255], # (cyan) lock
            [173, 216, 230], # (light blue) dam
            [255, 0, 255], # (magenta) solar farm
            [255, 165, 0], # (orange) power plant
            [128, 128, 0], # (olive) gas station
            [0, 255, 0], # (green) park
            [47, 79, 79], # (dark slate gray) parking garage
            [128, 0, 0], # (maroon) parking lot
            [165, 42, 42], # (brown) landfill
            [128, 128, 128], # (grey) quarry
            [255, 215, 0], # (gold) stadium
            [255, 105, 180], # (pink) airport
            [255, 255, 255], # (white) airport_runway
            [192, 192, 192], # (light grey) airport_taxiway
            [128, 0, 128], # (purple) airport_apron
            [0, 128, 0], # (dark green) airport_hangar
            [248, 248, 255], # (ghost white) airstrip
            [240, 230, 140], # (khaki) airport_terminal
            [192, 192, 192], # (silver) ski_resort
            [0, 96, 0], # (dark green) theme_park
            [95, 158, 160], # (cadet blue) storage_tank
            [205, 133, 63], # (peru) silo
            [154, 205, 50], # (yellow green) track
            [160, 82, 45], # (sienna) raceway
            [218, 112, 214], # (orchid) wastewater_plant
            [255, 255, 255], # (white) road
            [144, 238, 144], # (light green) railway
            [0, 0, 255], # (blue) river
            [255, 240, 245], # (lavender blush) water_park
            [65, 105, 225], # (royal blue) pier
            [238, 130, 238], # (violet) water_tower
            [75, 0, 130], # (indigo) street_lamp
            [233, 150, 122], # (dark salmon) traffic_signals
            [255, 255, 0], # (yellow) power_tower
            [255, 255, 0], # (yellow) power_substation
            [255, 0, 0], # (red) building
            [64, 64, 64], # (dark grey) bridge
            [255, 255, 255], # (white) road_motorway
            [255, 255, 255], # (white) road_trunk
            [255, 255, 255], # (white) road_primary
            [255, 255, 255], # (white) road_secondary
            [255, 255, 255], # (white) road_tertiary
            [255, 255, 255], # (white) road_residential
            [255, 255, 255], # (white) road_service
            [255, 255, 255], # (white) road_track
            [255, 255, 255], # (white) road_pedestrian
        ],
    },
    'land_cover': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': [
            'background',
            'water', 'developed', 'tree', 'shrub', 'grass',
            'crop', 'bare', 'snow', 'wetland', 'mangroves', 'moss',
        ],
        'colors': [
            [0, 0, 0], # unknown
            [0, 0, 255], # (blue) water
            [255, 0, 0], # (red) developed
            [0, 192, 0], # (dark green) tree
            [200, 170, 120], # (brown) shrub
            [0, 255, 0], # (green) grass
            [255, 255, 0], # (yellow) crop
            [128, 128, 128], # (grey) bare
            [255, 255, 255], # (white) snow
            [0, 255, 255], # (cyan) wetland
            [255, 0, 255], # (pink) mangroves
            [128, 0, 128], # (purple) moss
        ],
    },
    'tree_cover': {
        'type': 'regress',
        'BackgroundInvalid': True,
    },
    'crop_type': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': [
            'invalid',
            'rice', 'grape', 'corn', 'sugarcane',
            'tea', 'hop', 'wheat', 'soy', 'barley',
            'oats', 'rye', 'cassava', 'potato', 'sunflower', 'asparagus', 'coffee',
        ],
        'colors': [
            [0, 0, 0], # unknown
            [0, 0, 255], # (blue) rice
            [255, 0, 0], # (red) grape
            [255, 255, 0], # (yellow) corn
            [0, 255, 0], # (green) sugarcane
            [128, 0, 128], # (purple) tea
            [255, 0, 255], # (pink) hop
            [0, 128, 0], # (dark green) wheat
            [255, 255, 255], # (white) soy
            [128, 128, 128], # (grey) barley
            [165, 42, 42], # (brown) oats
            [0, 255, 255], # (cyan) rye
            [128, 0, 0], # (maroon) cassava
            [173, 216, 230], # (light blue) potato
            [128, 128, 0], # (olive) sunflower
            [0, 128, 0], # (dark green) asparagus
            [92, 64, 51], # (dark brown) coffee
        ],
    },
    'point': {
        'type': 'detect',
        'categories': [
            'background',
            'wind_turbine', 'lighthouse', 'mineshaft', 'aerialway_pylon', 'helipad',
            'fountain', 'toll_booth', 'chimney', 'communications_tower',
            'flagpole', 'petroleum_well', 'water_tower',
            'offshore_wind_turbine', 'offshore_platform', 'power_tower',
        ],
        'colors': [
            [0, 0, 0],
            [0, 255, 255], # (cyan) wind_turbine
            [0, 255, 0], # (green) lighthouse
            [255, 255, 0], # (yellow) mineshaft
            [0, 0, 255], # (blue) pylon
            [173, 216, 230], # (light blue) helipad
            [128, 0, 128], # (purple) fountain
            [255, 255, 255], # (white) toll_booth
            [0, 128, 0], # (dark green) chimney
            [128, 128, 128], # (grey) communications_tower
            [165, 42, 42], # (brown) flagpole
            [128, 0, 0], # (maroon) petroleum_well
            [255, 165, 0], # (orange) water_tower
            [255, 255, 0], # (yellow) offshore_wind_turbine
            [255, 0, 0], # (red) offshore_platform
            [255, 0, 255], # (magenta) power_tower
        ],
    },
    'rooftop_solar_panel': {
        'type': 'detect',
        'categories': [
            'background',
            'rooftop_solar_panel',
        ],
        'colors': [
            [0, 0, 0],
            [255, 255, 0], # (yellow) rooftop_solar_panel
        ],
    },
    'building': {
        'type': 'instance',
        'categories': [
            'background',
            'ms_building',
        ],
        'colors': [
            [0, 0, 0],
            [255, 255, 0], # (yellow) building
        ],
    },
    'polygon': {
        'type': 'instance',
        'categories': [
            'background',
            'aquafarm', 'lock', 'dam', 'solar_farm', 'power_plant', 'gas_station',
            'park', 'parking_garage', 'parking_lot', 'landfill', 'quarry', 'stadium',
            'airport', 'airport_apron', 'airport_hangar', 'airport_terminal',
            'ski_resort', 'theme_park', 'storage_tank', 'silo', 'track',
            'wastewater_plant', 'power_substation', 'pier', 'crop',
            'water_park',
        ],
        'colors': [
            [0, 0, 0],
            [255, 255, 0], # (yellow) aquafarm
            [0, 255, 255], # (cyan) lock
            [0, 255, 0], # (green) dam
            [0, 0, 255], # (blue) solar_farm
            [255, 0, 0], # (red) power_plant
            [128, 0, 128], # (purple) gas_station
            [255, 255, 255], # (white) park
            [0, 128, 0], # (dark green) parking_garage
            [128, 128, 128], # (grey) parking_lot
            [165, 42, 42], # (brown) landfill
            [128, 0, 0], # (maroon) quarry
            [255, 165, 0], # (orange) stadium
            [255, 105, 180], # (pink) airport
            [192, 192, 192], # (silver) airport_apron
            [173, 216, 230], # (light blue) airport_hangar
            [32, 178, 170], # (light sea green) airport_terminal
            [255, 0, 255], # (magenta) ski_resort
            [128, 128, 0], # (olive) theme_park
            [47, 79, 79], # (dark slate gray) storage_tank
            [255, 215, 0], # (gold) silo
            [192, 192, 192], # (light grey) track
            [240, 230, 140], # (khaki) wastewater_plant
            [154, 205, 50], # (yellow green) power_substation
            [255, 165, 0], # (orange) pier
            [0, 192, 0], # (middle green) crop
            [0, 192, 0], # (middle green) water_park
        ],
    },
    'wildfire': {
        'type': 'bin_segment',
        'categories': ['fire_retardant', 'burned'],
        'colors': [
            [255, 0, 0], # (red) fire retardant
            [128, 128, 128], # (grey) burned area
        ],
    },
    'smoke': {
        'type': 'classification',
        'categories': ['no', 'partial', 'yes'],
    },
    'snow': {
        'type': 'classification',
        'categories': ['no', 'partial', 'yes'],
    },
    'dem': {
        'type': 'regress',
        'BackgroundInvalid': True,
    },
    'airplane': {
        'type': 'detect',
        'categories': ['background', 'airplane'],
        'colors': [
            [0, 0, 0], # (black) background
            [255, 0, 0], # (red) airplane
        ],
    },
    'vessel': {
        'type': 'detect',
        'categories': ['background', 'vessel'],
        'colors': [
            [0, 0, 0], # (black) background
            [255, 0, 0], # (red) vessel
        ],
    },
    'water_event': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': ['invalid', 'background', 'water_event'],
        'colors': [
            [0, 0, 0], # (black) invalid
            [0, 255, 0], # (green) background
            [0, 0, 255], # (blue) water_event
        ],
    },
    'park_sport': {
        'type': 'classification',
        'categories': ['american_football', 'badminton', 'baseball', 'basketball', 'cricket', 'rugby', 'soccer', 'tennis', 'volleyball'],
    },
    'park_type': {
        'type': 'classification',
        'categories': ['park', 'pitch', 'golf_course', 'cemetery'],
    },
    'power_plant_type': {
        'type': 'classification',
        'categories': ['oil', 'nuclear', 'coal', 'gas'],
    },
    'quarry_resource': {
        'type': 'classification',
        'categories': ['sand', 'gravel', 'clay', 'coal', 'peat'],
    },
    'track_sport': {
        'type': 'classification',
        'categories': ['running', 'cycling', 'horse'],
    },
    'road_type': {
        'type': 'classification',
        'categories': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'track', 'pedestrian'],
    },
    'cloud': {
        'type': 'bin_segment',
        'categories': ['background', 'cloud', 'shadow'],
        'colors': [
            [0, 255, 0], # (green) not clouds or shadows
            [255, 255, 255], # (white) clouds
            [128, 128, 128], # (grey) shadows
	    ],
        'BackgroundInvalid': True,
    },
    'flood': {
        'type': 'bin_segment',
        'categories': ['background', 'water'],
        'colors': [
            [0, 255, 0], # (green) background
            [0, 0, 255], # (blue) water
        ],
        'BackgroundInvalid': True,
    },
}
