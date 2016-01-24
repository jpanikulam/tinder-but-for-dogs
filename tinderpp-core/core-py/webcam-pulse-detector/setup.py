from setuptools import setup

setup(
    name='webcam-pulse-detect',
    version='1.0',
    description='webcam pulse detection',
    long_description="Made by Thearn",
    author='Thearn',
    author_email='Thearn@github.com',
    url='https://github.com/thearn/webcam-pulse-detector',
    license='Apache 2.0',
    packages=['pulse_detect'],
    package_dir={'pulse_detect': 'pulse_detect'},
)

# APP = ['get_pulse.py']
# DATA_FILES = ['cascades/haarcascade_frontalface_alt.xml']
# OPTIONS = {'argv_emulation': True}

# setup(
    # app=APP,
    # data_files=DATA_FILES,
    # options={'py2app': OPTIONS},
    # setup_requires=['py2app'],
# )
