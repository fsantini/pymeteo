# pymeteo
App to display the typical meteo conditions at a location.

The app derives its data from meteostat.

## Tidal currents

The "Show tidal currents" option uses [pyTMD](https://pytmd.readthedocs.io). Current predictions need a tide model
with current constituents (e.g. TPXO9-atlas, which requires free registration, or FES2014 from AVISO). Download it
(see pyTMD's `fetch_box_tpxo` / `fetch_aviso_fes` tools) into `~/.cache/pytmd`, or set `PYTMD_DIR` to its location.

The "NOAA CO-OPS API" source needs no setup (US waters only, nearest prediction station within 30 km). TPXO9 and FES only appear in the source dropdown when their model files are installed.
