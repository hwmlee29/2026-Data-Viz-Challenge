#The script combines four sources to construct the dairy market pathways index and export a map and Excel table.

from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Folder structure used by this script:
#   raw/     source files read and processed by this script
#   outputs/ final map and county-level analysis table
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

FAME_FILE = RAW_DIR / "fame_master_feb2026.csv"
FOODHUB_FILE = RAW_DIR / "foodhub_2026.xlsx"
DAIRY_COWS_FILE = RAW_DIR / "dairy_cows_quickstats_2022.csv"
COUNTY_BOUNDARY_FILE = RAW_DIR / "county_boundaries.zip"

MAP_OUTPUT = OUTPUT_DIR / "Dairy_DataVisualization1.png"
TABLE_OUTPUT = OUTPUT_DIR / "Dairy_DataVisualization1.xlsx"

REQUIRED_INPUTS = [
    FAME_FILE,
    FOODHUB_FILE,
    DAIRY_COWS_FILE,
    COUNTY_BOUNDARY_FILE,
]

def check_required_inputs() -> None:
    missing = [path.name for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        file_list = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            "Missing required raw input file(s). Keep these files in the raw "
            f"folder next to local_market_channel_map.py:\n{file_list}"
        )

#This section defines the required input and output files and flags with a message if any of the source files is missing.

# ---------------------------------------------------------------------
# Final index specification
# ---------------------------------------------------------------------

DAIRY_THRESHOLD = 100      # Keep counties with at least 100 milk cows
FOODHUB_RADIUS_MI = 75     # Count local dairy food hubs within 75 miles
COLOR_CAP_Q = 0.95         # Cap map colors at the 95th percentile only

INDEX_COL = "local_market_channel_index"
FOODHUB_COUNT_COL = f"local_dairy_foodhubs_within_{FOODHUB_RADIUS_MI}mi"

UPPER_MIDWEST = ["MI", "MN", "WI"]
NORTHEAST = ["NY", "PA", "VT"]

CONTIGUOUS_STATES = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
    "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

#This section ensures conditions for which a county enters the analysis. A county enters only if it has at least 100 milk cows. The section also counts the number of dairy hubs within 75 miles of each county's representative point. This section also ensures that the map's color scale stops at 95th percentile (it does not change the calculated index) so that we can prevent extreme scores from dominating the map colors. Final visualization shows only six states, but we load contiguous United States because the national set defines the comparison universe for calculating z-scores.

# ---------------------------------------------------------------------
# Figure styling
# ---------------------------------------------------------------------

COLORS = {
    "background": "#FCFCFD",
    "missing": "#ECEFF3",
    "county_edge": "#FFFFFF",
    "state_edge": "#9BA4B5",
    "title_text": "#252A34",
    "legend_text": "#6F768A",
}

ORANGE_RAMP = mcolors.LinearSegmentedColormap.from_list(
    "orange_index",
    ["#F8E3B5", "#FBC96D", "#F48804", "#C44E00", "#742100"],
)
ORANGE_RAMP.set_over("#4D1300")

#The part above is used for colors and styling. We ensure that we use color-blind safe palettes here.

def zscore(values: pd.Series, universe: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce") 
    reference = numeric.loc[universe].dropna()
    output = pd.Series(np.nan, index=values.index, dtype=float)

    sd = reference.std(ddof=0)
    if reference.empty or sd == 0 or pd.isna(sd):
        return output

    output.loc[universe] = (numeric.loc[universe] - reference.mean()) / sd
    return output

#This section defines how to calculate the standardized z-score since variables used in the index are different and not directly comparable, so the z-score puts them in a comparable scale before they are averaged into the index. Specifically, it specifies which counties enter the analysis and how we handle missing or restricted values.


def complete_mean(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    complete = data[columns].notna().all(axis=1)
    output = pd.Series(np.nan, index=data.index, dtype=float)
    output.loc[complete] = data.loc[complete, columns].mean(axis=1)
    return output

#This section above defines how the three z-scores are used to create the index. It checks if every z-score is available for each county and for counties with all components, the average across columns is used as the index. We do not impute missing data in any county.

def load_counties() -> gpd.GeoDataFrame:
    counties = gpd.read_file(f"zip://{COUNTY_BOUNDARY_FILE}")
    counties = counties[counties["STUSPS"].isin(CONTIGUOUS_STATES)].copy()
    return counties[["GEOID", "NAME", "STUSPS", "geometry"]].to_crs("EPSG:5070")

#Here we load the boundaries since we need to put this index on a map. We use the national county set because the z-score reference values are calculated using qualifying dairy counties across the contiguous US, but we map only six states in the final visualization. Because we are using distance in our index, we project to EPSG:5070 for a safer distance calculations. 


def load_fame_variables() -> pd.DataFrame:
    fields = ["d2c_sales_pct", "intermediated_sales_pct"]
    fame = pd.read_csv(
        FAME_FILE,
        usecols=["fips", "county_name", "state_abbrev", "year", "variable_name", "value"],
    )
    fame = fame[fame["state_abbrev"].isin(CONTIGUOUS_STATES)]
    fame = fame[fame["year"].eq(2022) & fame["variable_name"].isin(fields)]
    wide = (
        fame.pivot_table(
            index=["fips", "county_name", "state_abbrev"],
            columns="variable_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide["GEOID"] = wide["fips"].astype(int).astype(str).str.zfill(5)
    return wide[["GEOID", *fields]]

#This above section reads the FAME file, keeps the 2022 county records for the two measures we specify (direct to consumer and intermediated sales), reshapes them into one row per county and returns the components needed for the index calculation.


def load_dairy_base() -> pd.DataFrame:
    data = pd.read_csv(
        DAIRY_COWS_FILE,
        dtype={"State ANSI": str, "County ANSI": str},
        low_memory=False,
    )
    data = data[
        data["Year"].eq(2022)
        & data["Geo Level"].eq("COUNTY")
        & data["Data Item"].eq("CATTLE, COWS, MILK - INVENTORY")
    ].copy()
    data = data[data[["State ANSI", "County ANSI"]].notna().all(axis=1)].copy()

    data["GEOID"] = data["State ANSI"].str.zfill(2) + data["County ANSI"].str.zfill(3)
    data["value_numeric"] = pd.to_numeric(
        data["Value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    inventory = (
        data[data["Domain"].eq("TOTAL")]
        .groupby("GEOID", as_index=False)["value_numeric"]
        .first()
        .rename(columns={"value_numeric": "cow_inventory_2022"})
    )

    class_rows = data[data["Domain"].eq("INVENTORY OF MILK COWS")].copy()
    class_rows["class_lower"] = pd.to_numeric(
        class_rows["Domain Category"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"\((\d+)\s+(?:TO|OR MORE)")[0],
        errors="coerce",
    )
    class_lower = (
        class_rows.groupby("GEOID", as_index=False)["class_lower"]
        .max()
        .rename(columns={"class_lower": "max_reported_inventory_class_lower"})
    )

    data = inventory.merge(class_lower, on="GEOID", how="outer")
    inventory = pd.to_numeric(data["cow_inventory_2022"], errors="coerce")
    class_lower = pd.to_numeric(data["max_reported_inventory_class_lower"], errors="coerce")

    data["dairy_base_100plus"] = inventory.ge(DAIRY_THRESHOLD) | class_lower.ge(DAIRY_THRESHOLD)
    data["dairy_base_100plus"] = data["dairy_base_100plus"].fillna(False)

    keep = [
        "GEOID",
        "cow_inventory_2022",
        "max_reported_inventory_class_lower",
        "dairy_base_100plus",
    ]
    return data[keep]

#This section ensures that counties with at least 100 cows enter the analysis. In more detail, it uses reported number of cows for those counties which have been published. However, for counties where this information has been suppressed, we use the published class (100+ cows, example 100-199 milk cows) to make sure they are included in the analysis.

def load_foodhubs() -> gpd.GeoDataFrame:
    hubs = pd.read_excel(FOODHUB_FILE, sheet_name="Data")
    local_dairy = hubs["productslocality_dairyproducts"].astype(str).str.contains(
        "Exclusively local|Both local", case=False, na=False
    )
    hubs = hubs[local_dairy & hubs[["location_x", "location_y"]].notna().all(axis=1)].copy()

    return gpd.GeoDataFrame(
        hubs,
        geometry=gpd.points_from_xy(hubs["location_x"], hubs["location_y"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")


def add_foodhub_count(counties: gpd.GeoDataFrame, hubs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    counties = counties.copy()
    county_points = counties.geometry.representative_point()
    county_xy = np.column_stack([county_points.x, county_points.y])
    hub_xy = np.column_stack([hubs.geometry.x, hubs.geometry.y])

    if len(hub_xy) == 0:
        counties[FOODHUB_COUNT_COL] = 0
        return counties

    distances_mi = (
        np.sqrt(((county_xy[:, None, :] - hub_xy[None, :, :]) ** 2).sum(axis=2)) / 1609.344
    )
    counties[FOODHUB_COUNT_COL] = (distances_mi <= FOODHUB_RADIUS_MI).sum(axis=1)
    return counties

#The above section keeps food hubs that source local dairy and then calculates the distance from each county's interior point, generated by GeoPandas' representative_point() method, to each retained food hub, so that hubs within a 75-mile radius can be counted for use in the index.


def build_county_data() -> gpd.GeoDataFrame:
    counties = load_counties()
    fame = load_fame_variables()
    dairy = load_dairy_base()
    hubs = load_foodhubs()

    data = counties.merge(fame, on="GEOID", how="left")
    data = data.merge(dairy, on="GEOID", how="left")
    data["dairy_base_100plus"] = data["dairy_base_100plus"].fillna(False)
    data = add_foodhub_count(data, hubs)

    dairy_universe = data["dairy_base_100plus"].astype(bool)

    data["z_d2c_sales_pct"] = zscore(data["d2c_sales_pct"], dairy_universe)
    data["z_intermediated_sales_pct"] = zscore(data["intermediated_sales_pct"], dairy_universe)
    data["z_foodhub_count"] = zscore(data[FOODHUB_COUNT_COL], dairy_universe)

    data[INDEX_COL] = complete_mean(
        data,
        ["z_d2c_sales_pct", "z_intermediated_sales_pct", "z_foodhub_count"],
    )
    return data

#This above section combines all the counties that qualify using the 100 milk cow threshold as well as the other data at the county level to be used for the index calculation, it averages the index, and provides a complete dataset to be used for mapping the index and generating the Excel file.

def color_limits(data: gpd.GeoDataFrame) -> tuple[float, float]:
    values = data[INDEX_COL].dropna()
    lower = np.floor(values.min() * 10) / 10
    upper = np.ceil(values.quantile(COLOR_CAP_Q) * 10) / 10
    return float(lower), float(upper)

#Here we create the map color range by enforcing the cap at 95th percentile so extreme outliers do not dominate the colors.


def export_analysis_table(data: gpd.GeoDataFrame) -> None:
    _, color_cap = color_limits(data)
    output = pd.DataFrame(data.drop(columns="geometry"))
    output["index_above_color_cap_95pct"] = output[INDEX_COL].gt(color_cap)
    output["mapped_final_index"] = output[INDEX_COL].notna()

    columns = [
        "GEOID",
        "NAME",
        "STUSPS",
        "cow_inventory_2022",
        "max_reported_inventory_class_lower",
        "dairy_base_100plus",
        "d2c_sales_pct",
        "intermediated_sales_pct",
        FOODHUB_COUNT_COL,
        "z_d2c_sales_pct",
        "z_intermediated_sales_pct",
        "z_foodhub_count",
        INDEX_COL,
        "index_above_color_cap_95pct",
        "mapped_final_index",
    ]
    output[columns].sort_values(["STUSPS", "NAME", "GEOID"]).to_excel(
        TABLE_OUTPUT, index=False
    )

#This section above produces the Excel file to be used for mapping as well as for the actual index score for the counties included.

def draw_panel(
    ax: plt.Axes,
    data: gpd.GeoDataFrame,
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    data.plot(
        ax=ax,
        color=COLORS["missing"],
        edgecolor=COLORS["county_edge"],
        linewidth=0.06,
        zorder=1,
    )

    mapped = data[data[INDEX_COL].notna()].copy()
    mapped.plot(
        ax=ax,
        column=INDEX_COL,
        cmap=ORANGE_RAMP,
        vmin=vmin,
        vmax=vmax,
        edgecolor=COLORS["county_edge"],
        linewidth=0.06,
        zorder=2,
    )

    data.dissolve(by="STUSPS").boundary.plot(
        ax=ax,
        color=COLORS["state_edge"],
        linewidth=0.55,
        zorder=3,
    )

    minx, miny, maxx, maxy = data.total_bounds
    ax.set_xlim(minx - (maxx - minx) * 0.015, maxx + (maxx - minx) * 0.015)
    ax.set_ylim(miny - (maxy - miny) * 0.02, maxy + (maxy - miny) * 0.02)
    ax.set_axis_off()
    ax.set_title(
        title,
        fontsize=12.5,
        fontweight="semibold",
        color=COLORS["title_text"],
        pad=6,
    )

#The section above formats the map to be more public-facing, removing clutter, ensures that the defined map color range, ensures that counties with missing index are shown in gray, adds the state boundaries.

def draw_legend(fig: plt.Figure, vmin: float, vmax: float) -> None:
    cax = fig.add_axes([0.36, 0.06, 0.36, 0.022])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=ORANGE_RAMP),
        cax=cax,
        orientation="horizontal",
        extend="max",
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    colorbar.ax.tick_params(labelsize=8.5, colors=COLORS["legend_text"], length=0)

#This section above generates the legend.

def export_map(data: gpd.GeoDataFrame) -> None:
    vmin, vmax = color_limits(data)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.8), dpi=180)
    fig.patch.set_facecolor(COLORS["background"])

    draw_panel(
        axes[0],
        data[data["STUSPS"].isin(UPPER_MIDWEST)].copy(),
        vmin,
        vmax,
        "Midwest (MI, MN, WI)",
    )
    draw_panel(
        axes[1],
        data[data["STUSPS"].isin(NORTHEAST)].copy(),
        vmin,
        vmax,
        "Northeast (NY, PA, VT)",
    )

    fig.text(
        0.5,
        0.955,
        "Local-Market Channels in Dairy-Producing Counties",
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=COLORS["title_text"],
    )
    fig.text(
        0.5,
        0.115,
        "Z-score index of direct-to-consumer sales, intermediated local-channel sales, "
        "and local dairy food hubs within 75 miles among 100+ milk-cow counties",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLORS["legend_text"],
    )
    draw_legend(fig, vmin, vmax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.16, wspace=0.03)
    fig.savefig(MAP_OUTPUT, dpi=300, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close(fig)

#This section above draws the two panels, adds title, index description, legend, and then it exports it to a PNG file.


def main() -> None:
    check_required_inputs()
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = build_county_data()
    export_analysis_table(data)
    export_map(data)
    print(f"Created {TABLE_OUTPUT.name}")
    print(f"Created {MAP_OUTPUT.name}")


if __name__ == "__main__":
    main()
