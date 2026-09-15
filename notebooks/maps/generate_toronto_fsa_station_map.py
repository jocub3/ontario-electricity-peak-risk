from __future__ import annotations

from pathlib import Path
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


OUTPUT_DIR = Path("map_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_FSAS = ["L4T", "M5R", "M5S", "M6G", "M9R", "M9W"]

STATIONS = pd.DataFrame(
    [
        {"station": "Toronto City", "climate_id": "6158355", "latitude": 43.6667, "longitude": -79.4000},
        {"station": "Toronto INTL A", "climate_id": "6158731", "latitude": 43.6767, "longitude": -79.6306},
    ]
)

FSA_TO_STATION = {
    "M5S": "Toronto City",
    "M5R": "Toronto City",
    "M6G": "Toronto City",
    "L4T": "Toronto INTL A",
    "M9W": "Toronto INTL A",
    "M9R": "Toronto INTL A",
}

COVERAGE_COLORS = {
    "Toronto City": "#4C78A8",
    "Toronto INTL A": "#F28E2B",
}

OTHER_FSA_FACE = "#F2F2F2"
OTHER_FSA_EDGE = "#CFCFCF"
ONTARIO_FACE = "#EFEFEF"
ONTARIO_EDGE = "#777777"
TEXT_COLOR = "#222222"

BASE_URL = (
    "https://geo.statcan.gc.ca/geo_wa/rest/services/"
    "2021/Cartographic_boundary_files/MapServer"
)

FSA_LAYER_URL = f"{BASE_URL}/14/query"
CMA_LAYER_URL = f"{BASE_URL}/6/query"
PROVINCE_LAYER_URL = f"{BASE_URL}/0/query"

MAP_CRS = "EPSG:3347"
TORONTO_CMA_UID = "535"
ONTARIO_PRUID = "35"


def download_geojson(url: str, params: dict, output_path: Path) -> None:
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    geojson = response.json()

    if "features" not in geojson or len(geojson["features"]) == 0:
        raise RuntimeError(
            "Statistics Canada returned no features.\n"
            f"Request URL: {response.url}"
        )

    output_path.write_text(json.dumps(geojson), encoding="utf-8")
    print(f"Downloaded {len(geojson['features'])} feature(s): {output_path}")


def load_or_download_all_ontario_fsas() -> gpd.GeoDataFrame:
    path = OUTPUT_DIR / "ontario_fsas_statcan_2021.geojson"

    if not path.exists():
        params = {
            "where": f"PRUID = '{ONTARIO_PRUID}'",
            "outFields": "CFSAUID,PRUID,PRNAME,LANDAREA",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        download_geojson(FSA_LAYER_URL, params, path)

    gdf = gpd.read_file(path)
    gdf["CFSAUID"] = gdf["CFSAUID"].astype(str)
    return gdf


def load_or_download_toronto_cma() -> gpd.GeoDataFrame:
    path = OUTPUT_DIR / "toronto_cma_statcan_2021.geojson"

    if not path.exists():
        params = {
            "where": f"CMAUID = '{TORONTO_CMA_UID}'",
            "outFields": "CMAUID,CMANAME,CMATYPE",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        download_geojson(CMA_LAYER_URL, params, path)

    return gpd.read_file(path)


def load_or_download_ontario() -> gpd.GeoDataFrame:
    path = OUTPUT_DIR / "ontario_statcan_2021.geojson"

    if not path.exists():
        params = {
            "where": f"PRUID = '{ONTARIO_PRUID}'",
            "outFields": "PRUID,PRNAME",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        download_geojson(PROVINCE_LAYER_URL, params, path)

    return gpd.read_file(path)


def build_station_geodataframe() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        STATIONS.copy(),
        geometry=gpd.points_from_xy(
            STATIONS["longitude"],
            STATIONS["latitude"],
        ),
        crs="EPSG:4326",
    )


def select_fsas_intersecting_cma(
    ontario_fsas: gpd.GeoDataFrame,
    toronto_cma: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    fsas = ontario_fsas.to_crs(MAP_CRS)
    cma = toronto_cma.to_crs(MAP_CRS)

    cma_union = cma.geometry.union_all()

    return fsas.loc[
        fsas.geometry.intersects(cma_union)
    ].copy()


def add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.045, 0.86),
        xytext=(0.045, 0.75),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops={
            "facecolor": "black",
            "edgecolor": "black",
            "width": 5,
            "headwidth": 14,
            "headlength": 16,
        },
        zorder=20,
    )


def add_scale_bar(ax, length_km: float = 30) -> None:
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()

    length_m = length_km * 1000
    x0 = minx + (maxx - minx) * 0.04
    y0 = miny + (maxy - miny) * 0.06
    segment = length_m / 3

    for i in range(3):
        ax.add_patch(
            Rectangle(
                (x0 + i * segment, y0),
                segment,
                (maxy - miny) * 0.008,
                facecolor="black" if i % 2 == 0 else "white",
                edgecolor="black",
                linewidth=0.8,
                zorder=20,
            )
        )

    ax.text(
        x0,
        y0 + (maxy - miny) * 0.018,
        "0",
        fontsize=8,
        ha="center",
        va="bottom",
    )

    ax.text(
        x0 + length_m,
        y0 + (maxy - miny) * 0.018,
        f"{int(length_km)} km",
        fontsize=8,
        ha="center",
        va="bottom",
    )


def create_map(output_path: Path) -> None:
    ontario_fsas = load_or_download_all_ontario_fsas()
    toronto_cma = load_or_download_toronto_cma()
    ontario = load_or_download_ontario()
    stations = build_station_geodataframe()

    fsas_context = select_fsas_intersecting_cma(
        ontario_fsas,
        toronto_cma,
    )

    cma_plot = toronto_cma.to_crs(MAP_CRS)
    ontario_plot = ontario.to_crs(MAP_CRS)
    stations_plot = stations.to_crs(MAP_CRS)

    fsas_context["assigned_station"] = (
        fsas_context["CFSAUID"].map(FSA_TO_STATION)
    )

    study_fsas = fsas_context.loc[
        fsas_context["CFSAUID"].isin(STUDY_FSAS)
    ].copy()

    missing = sorted(set(STUDY_FSAS) - set(study_fsas["CFSAUID"]))

    if missing:
        raise RuntimeError(
            f"Study FSAs missing from Toronto-region data: {missing}"
        )

    fig, ax = plt.subplots(figsize=(15, 10))

    cma_plot.plot(
        ax=ax,
        facecolor="#FAFAFA",
        edgecolor="#A0A0A0",
        linewidth=1.0,
        zorder=0,
    )

    fsas_context.plot(
        ax=ax,
        facecolor=OTHER_FSA_FACE,
        edgecolor=OTHER_FSA_EDGE,
        linewidth=0.45,
        alpha=1.0,
        zorder=1,
    )

    for station_name, color in COVERAGE_COLORS.items():
        subset = study_fsas.loc[
            study_fsas["assigned_station"] == station_name
        ]

        subset.plot(
            ax=ax,
            facecolor=color,
            edgecolor="white",
            linewidth=1.3,
            alpha=0.68,
            zorder=4,
        )

        subset.boundary.plot(
            ax=ax,
            color=color,
            linewidth=1.0,
            zorder=5,
        )

    for _, row in study_fsas.iterrows():
        p = row.geometry.representative_point()

        ax.text(
            p.x,
            p.y,
            row["CFSAUID"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            zorder=8,
        )

    for _, station in stations_plot.iterrows():
        color = COVERAGE_COLORS[station["station"]]

        ax.scatter(
            station.geometry.x,
            station.geometry.y,
            s=230,
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=1.0,
            zorder=10,
        )

    minx, miny, maxx, maxy = cma_plot.total_bounds

    xpad = (maxx - minx) * 0.035
    ypad = (maxy - miny) * 0.035

    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)

    legend_items = [
        Patch(
            facecolor=COVERAGE_COLORS["Toronto City"],
            edgecolor=COVERAGE_COLORS["Toronto City"],
            alpha=0.68,
            label="Toronto City coverage (M5S, M5R, M6G)",
        ),
        Patch(
            facecolor=COVERAGE_COLORS["Toronto INTL A"],
            edgecolor=COVERAGE_COLORS["Toronto INTL A"],
            alpha=0.68,
            label="Toronto INTL A coverage (L4T, M9W, M9R)",
        ),
        Line2D(
            [0], [0],
            marker="*",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=13,
            linestyle="None",
            label="Weather station",
        ),
        Patch(
            facecolor=OTHER_FSA_FACE,
            edgecolor=OTHER_FSA_EDGE,
            label="Other FSAs in Toronto CMA",
        ),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper left",
        frameon=True,
        framealpha=0.96,
        title="Legend",
        fontsize=9,
    )

    add_north_arrow(ax)
    add_scale_bar(ax, length_km=30)

    inset = fig.add_axes([0.73, 0.62, 0.23, 0.25])

    ontario_plot.plot(
        ax=inset,
        facecolor=ONTARIO_FACE,
        edgecolor=ONTARIO_EDGE,
        linewidth=0.6,
    )

    cma_plot.plot(
        ax=inset,
        facecolor="#666666",
        edgecolor="#444444",
        linewidth=0.5,
        alpha=0.90,
    )

    inset.set_title(
        "Toronto CMA within Ontario",
        fontsize=9,
        pad=4,
    )
    inset.set_axis_off()

    ax.set_title(
        "Ontario (Toronto Region) Forward Sortation Areas (FSAs) and Weather Stations",
        fontsize=16,
        fontweight="semibold",
        pad=14,
    )

    ax.set_axis_off()

    fig.text(
        0.01,
        0.012,
        (
            "Source: Statistics Canada, 2021 Census Cartographic Boundary Files "
            "(FSA, CMA and Province/Territory layers)."
        ),
        fontsize=8,
        color="#555555",
        ha="left",
    )

    fig.text(
        0.01,
        0.002,
        "Map projection: NAD83 / Statistics Canada Lambert (EPSG:3347).",
        fontsize=8,
        color="#555555",
        ha="left",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print(f"Map saved to: {output_path.resolve()}")


if __name__ == "__main__":
    create_map(
        OUTPUT_DIR
        / "Toronto_FSA_Weather_Stations_Presentation_Map.png"
    )
