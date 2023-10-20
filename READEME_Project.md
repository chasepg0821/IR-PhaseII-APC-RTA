# Run the author's original code

## Installation

### Code

```
git clone https://github.com/deezer/APC-RTA
cd APC-RTA
pip install -r requirements.txt
```

Requirements: implicit==0.6.1, matplotlib==3.6.2, pandas==1.5.2, psutil==5.9.4, pympler==1.0.1, scipy==1.7.3, seaborn==0.12.1, tables==3.7.0, tqdm==4.64.1.

### Data

Please download Spotify's Million Playlist Dataset (MPD) on [AIcrowd.com](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).

You would need to create an account and register to the challenge to do so.

<p align="center">
  <img height="325" src="figures/mpd.png">
</p>

Then, please unzip all files in a single folder, for instance: `resources/data/raw_MPD`.

Run the following script to pre-process and format the MPD (expected time: around 1 hour on a regular laptop).

```
python src/format_rta_input --mpd_path PATH/TO/UNZIPPED/MPD
```

# Run OUR code

## Installation

### Data

We used the same dataset as the authors the MPD dataset from Spotify

Download the dataset from [AIcrowd.com](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).

Unzip the files.

### Code

Import numpy, tensorflow, and tqdm.

We used the latest versions of each

From there you are able to simply run the file RTA-Project.py

Ensure you run the file from within the directory containing all of the slices of data