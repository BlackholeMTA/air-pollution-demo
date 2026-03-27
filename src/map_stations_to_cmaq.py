from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
from netCDF4 import Dataset

NC_FILE = Path("data/raw/cmaq/CCTM_ACONC_v532_gcc_v53_20251225.nc")
STATION_FILE = Path("data/raw/stations/station_metadata.csv")
OUT_FILE = Path("data/processed/station_cmaq_mapping.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def build_cmaq_transformer(ds):
    # Lambert Conformal Conic from CMAQ attrs
    p_alp = float(ds.P_ALP)
    p_bet = float(ds.P_BET)
    p_gam = float(ds.P_GAM)
    xcent = float(ds.XCENT)
    ycent = float(ds.YCENT)

    proj4 = (
        f"+proj=lcc +lat_1={p_alp} +lat_2={p_bet} "
        f"+lat_0={ycent} +lon_0={p_gam} "
        f"+datum=WGS84 +units=m +no_defs"
    )

    crs_cmaq = CRS.from_proj4(proj4)
    crs_wgs84 = CRS.from_epsg(4326)

    # đổi từ WGS84 (lon, lat) -> CMAQ projected x,y
    transformer = Transformer.from_crs(crs_wgs84, crs_cmaq, always_xy=True)
    return transformer


def main():
    ds = Dataset(NC_FILE)
    transformer = build_cmaq_transformer(ds)

    xor = float(ds.XORIG)
    yor = float(ds.YORIG)
    xcell = float(ds.XCELL)
    ycell = float(ds.YCELL)
    ncols = int(ds.NCOLS)
    nrows = int(ds.NROWS)

    stations = pd.read_csv(STATION_FILE)

    rows = []
    for _, row in stations.iterrows():
        station_id = row["station_id"]
        station_name = row["station_name"]
        lat = float(row["lat"])
        lon = float(row["lon"])

        # WGS84 -> projected x,y
        x, y = transformer.transform(lon, lat)

        # tính cột, hàng gần nhất
        col = int(round((x - xor) / xcell - 0.5))
        grid_row_from_south = int(round((y - yor) / ycell - 0.5))

        # CMAQ row index thường tính từ south->north theo hình học này
        row_idx = grid_row_from_south

        inside = (0 <= col < ncols) and (0 <= row_idx < nrows)

        rows.append({
            "station_id": station_id,
            "station_name": station_name,
            "lat": lat,
            "lon": lon,
            "x_proj": x,
            "y_proj": y,
            "row": row_idx,
            "col": col,
            "inside_domain": inside
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)

    print(out_df)
    print(f"\nSaved to {OUT_FILE}")

    ds.close()

if __name__ == "__main__":
    main()
