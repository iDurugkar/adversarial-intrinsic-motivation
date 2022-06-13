import os

from stable_baselines.her import HER
from stable_baselines.td3 import TD3, DDLTD3
from stable_baselines.aim_td3 import AIMTD3, GAILTD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_file, 'r') as file_handler:
    __version__ = file_handler.read().strip()
