from os import popen

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import random


# --- CONFIGURACION ---

ARCHIVO_GPKG = "DATASETS/DIVISIONES_ADMINISTRATIVAS_CM/DIVISIONES_ADMINISTRATIVAS_CM.gpkg"
ARCHIVOS_PARADAS = [
     "DATASETS/GTFS Red de Autobuses Interurbanos/stops.txt",
"DATASETS/GTFS Red de Autobuses Urbanos/stops.txt",
    "DATASETS/GTFS Red de Metro/stops.txt",
    "DATASETS/GTFS Red de EMT/stops.txt",
    "DATASETS/GTFS Red de Metro Ligero/stops.txt"
]

ARCHIVO_POB = "DATASETS/33847.csv"
list_dfs_stops = []
gdf_madrid_boundaries = None
list_pupulation_municipality = []


def fill_gdf_madrid_boundaries():

    global gdf_madrid_boundaries

    try:
        gdf_madrid_boundaries = gpd.read_file(ARCHIVO_GPKG, layer='IDEM_CM_UNID_ADMIN') #poligonos
        print(gdf_madrid_boundaries.head())
        gdf_madrid_boundaries = gdf_madrid_boundaries.to_crs(epsg=3857) # transformation to epsg 3857 standard for all of my maps.
        # as idem_cm_unid_admin is in EPSG:25830 (ETRS89 UTM huso 30)
        print(gdf_madrid_boundaries.head())
        print(gdf_madrid_boundaries.info())

        print("\n"+ "\n")

        print(f" downloaded map with : {len(gdf_madrid_boundaries)} lines")

    except Exception as e:
        print(f" Error {e}")
        exit()

def fill_stops():

    global list_dfs_stops
    x =0
    for archivo in ARCHIVOS_PARADAS:
        try:

            df = pd.read_csv(archivo)
            df = df[['stop_lat', 'stop_lon']].dropna() # cleaning the NA values, get rid of the other info
            gdf_stops = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.stop_lon, df.stop_lat),crs="EPSG:4326")
            gdf_stops = gdf_stops.to_crs(epsg=3857)
            list_dfs_stops.append(gdf_stops)
            if x < 1 :
                print("vis the gdf objects \n", str(gdf_stops.head()), "\n", type(gdf_stops), type(gdf_stops.geometry), type(gdf_stops.geometry.x) , "\n",str(gdf_stops.geometry), str(gdf_stops.geometry.x))

            print(f"  added : {archivo}")
            x += 1
        except Exception as e:
            print(f"  error adding {archivo}´: {e}")
    if not list_dfs_stops:
        print("No datasets")
        exit()


def fill_population():
    global list_pupulation_municipality
    try:
        # Changed separator to tab based on file inspection
        # Specifying dtype for Total to avoid DtypeWarning (mixed types due to thousands separators)
        df = pd.read_csv(ARCHIVO_POB, sep='\t', encoding='utf-8', dtype={'Total': str})
        df.columns = df.columns.str.strip()
        
        mask_sexo = df['Sexo'].astype(str).str.strip() == "Total"
        mask_fecha = df['Periodo'].astype(str).str.strip() == "1 de enero de 2022"
        df_filtered = df[mask_sexo & mask_fecha].copy()
        
        # Extract numeric age from 'Edad (año a año)'
        # Format is like "0 años", "1 año", "100 años y más", "Todas las edades"
        def extract_age(age_str):
            age_str = str(age_str).strip()
            if "Todas las edades" in age_str:
                return -1 # Filter out
            parts = age_str.split(' ')
            if parts[0].isdigit():
                return int(parts[0])
            return -1

        df_filtered['age_num'] = df_filtered['Edad (año a año)'].apply(extract_age)

        df_final = df_filtered[(df_filtered['age_num'] >= 18) & (df_filtered['age_num'] <= 24)].copy()
        
        def extract_mun_code(mun_str):
            return str(mun_str).split(' ')[0]
        def extract_mun_name(mun_str):
            parts = str(mun_str).split(' ', 1)
            if len(parts) > 1:
                return parts[1]
            return mun_str

        df_final['CD_INE'] = df_final['Municipios'].apply(extract_mun_code)
        df_final['DS_NOMBRE'] = df_final['Municipios'].apply(extract_mun_name)
        
        # Clean Total column (remove dots if present for thousands separator)
        df_final["Total"] = df_final["Total"].astype(str).str.replace('.', '', regex=False)
        df_final["Total"] = pd.to_numeric(df_final["Total"], errors='coerce')
        
        df_final = df_final.groupby(['CD_INE', 'DS_NOMBRE'])['Total'].sum().reset_index()
        list_pupulation_municipality = df_final

        print("::::::::::::::::::::::::::::::::::::::::::")
        print(list_pupulation_municipality.head())

        print("::::::::::::::::::::::::::::::::::::::::::::::::::::")

    except Exception as e:
        print(f"Error adding {ARCHIVO_POB}: {e}")
        import traceback
        traceback.print_exc()
        exit()
    

fill_stops()
fill_gdf_madrid_boundaries()
fill_population()

#::::::::::::::::::::::: CHOROPLETH :::::::::::::::::::::::::

gdf_choropleth = gdf_madrid_boundaries.merge(
    list_pupulation_municipality,
    left_on='CD_INE',
    right_on='CD_INE',
    how='left',
    suffixes=('', '_pop')
)

gdf_choropleth['Total'] = gdf_choropleth['Total'].fillna(0)

print("##############################################################")
print(gdf_choropleth.head())
print(gdf_choropleth.info())
print("##############################################################")


# ::::::::::::::::::::::::::: KDE :::::::::::::::::::::::::::

all_stops = pd.concat(list_dfs_stops)
x_stops = all_stops.geometry.x
y_stops = all_stops.geometry.y
xmin, ymin, xmax, ymax = gdf_madrid_boundaries.total_bounds

RESOLUTION = 400 #in how many pixeles we will observe the KDE
xx, yy = np.mgrid[xmin:xmax:complex(0, RESOLUTION), ymin:ymax:complex(0, RESOLUTION)] #two main matrix, [500 * 500] dividing the space in parts
positions = np.vstack([xx.ravel(), yy.ravel()]) # one "inline" array for xx and other for yy with all the divisions : [2, 50.000]
values = np.vstack([x_stops, y_stops])
#all the cordenates of the stops "inline"
print("Vis values"+ str(values[1,random.randint(0, len(values)-1)]), type(values), len(values))
print("vis positios"+ str(positions[1,random.randint(0, len(positions)-1)]), type(positions), len(positions))
# KDE
kernel = gaussian_kde(values, bw_method=0.05)#resultand function: sum of the all the gauss bells of all the stops

f = kernel(positions)
print("vis of f" + str(f.max()), type(f), type(f.max()))

f_norm = (f - f.min()) / (f.max() - f.min()) * 100  #as we have density prob expressions: 3.4e-05 we nned to normalize
f_log = np.log1p(f)
f_norm_log = (f_log - f_log.min()) / (f_log.max() - f_log.min()) * 100 #logaritmical normalization
f_grid = f_norm_log.reshape(xx.shape)
print("vis of f_norm_log" + str(f_norm.max()), type(f_norm))



#------ Generating the plot ----

fig, ax = plt.subplots(figsize=(12, 12))


gdf_madrid_boundaries.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)


cax = ax.imshow(np.rot90(f_grid), cmap='inferno', extent=[xmin, xmax, ymin, ymax],
                alpha=0.5, zorder=2, aspect='auto') #plotting the heatmap

ax.set_title("Transport Density in Madrid + base map", fontsize=15)

cbar = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Transport Density (0-100)')


plt.savefig("HeatMap.png", dpi=300, bbox_inches='tight')
print("Image saved as 'HeatMap.png'")


# ::::::::::::::::::::::: CHOROPLETH MAP :::::::::::::::::::::::::


fig2, ax2 = plt.subplots(figsize=(20, 20)) 
gdf_madrid_boundaries.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)

plot_choro = gdf_choropleth.plot(
    ax=ax2,
    column='Total',
    cmap='viridis',      
    legend=True,
    legend_kwds={'label': "Población Joven (18-24 años)", 'shrink': 0.6},
    edgecolor='black',
    linewidth=0.2
)

POPULATION_THRESHOLD = 100 
import matplotlib.patheffects as pe

texts = []
for idx, row in gdf_choropleth.iterrows():
    pop = row['Total']
    if pop > POPULATION_THRESHOLD:
        point = row.geometry.representative_point().coords[0]
        name = row['DS_NOMBRE'] if pd.notnull(row['DS_NOMBRE']) else ""
        label_text = f"{name}\n({int(pop)})"
        ax2.annotate(
            text=label_text,
            xy=point,
            ha='center',
            va='center',
            fontsize=6, 
            color='white',
            fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground="black")] # Outline for readability
        )

ax2.set_title("Población Joven (18-24) por Municipio", fontsize=20)
ax2.axis('off') 


plt.savefig("ChoroplethMap.png", dpi=300, bbox_inches='tight')
print("Image saved as 'ChoroplethMap.png'")

# ::::::::::::::::::::::: OPPORTUNITY SCORE MODEL :::::::::::::::::::::::::

print("Generating Opportunity Score Model...")


# Ensure CRS is projected (meters) for area calculation. It is already EPSG:3857.
gdf_choropleth['area_km2'] = gdf_choropleth.geometry.area / 10**6
gdf_choropleth['pop_density'] = gdf_choropleth['Total'] / gdf_choropleth['area_km2']


gdf_choropleth['pop_density_log'] = np.log1p(gdf_choropleth['pop_density']) # Normalize Population Density (Log Normalization 0-1)
# Add a small constant to avoid log(0)
min_pop = gdf_choropleth['pop_density_log'].min()
max_pop = gdf_choropleth['pop_density_log'].max()
gdf_choropleth['Pop_Score'] = (gdf_choropleth['pop_density_log'] - min_pop) / (max_pop - min_pop)
gdf_choropleth['Pop_Score'] = gdf_choropleth['Pop_Score'].fillna(0)


# We reuse xx, yy from the KDE step.
# Flatten the grid coordinates
x_flat = xx.ravel()
y_flat = yy.ravel()

# Create a GeoDataFrame for the grid points
gdf_grid = gpd.GeoDataFrame(
    {'geometry': gpd.points_from_xy(x_flat, y_flat)},
    crs="EPSG:3857"
)

# Perform Spatial Join to assign Municipality info to each grid point
# We only need the Pop_Score and Municipality Name
gdf_grid_joined = gpd.sjoin(
    gdf_grid, 
    gdf_choropleth[['geometry', 'DS_NOMBRE', 'Pop_Score']], 
    how='left', 
    predicate='within'
)

# Fill NaN values (points outside any municipality) with 0
gdf_grid_joined['Pop_Score'] = gdf_grid_joined['Pop_Score'].fillna(0)
gdf_grid_joined['DS_NOMBRE'] = gdf_grid_joined['DS_NOMBRE'].fillna("Unknown")

# --- Step 3: Calculate Final Opportunity Score ---


# Transport Score: f_grid is already calculated (0-100).
# We need it flattened to match the grid points.
# Note: f_grid comes from f_norm_log which is 0-100.
# The user wants: (1 - Transport_Score) * 0.4.
# So we need Transport_Score in 0-1 range.
transport_score_flat = f_grid.ravel() / 100.0 

# Pop Score is already 0-1.
pop_score_flat = gdf_grid_joined['Pop_Score'].values

# Formula: Final_Score = (Pop_Score * 0.6) + ((1 - Transport_Score) * 0.4)
final_score_flat = (pop_score_flat * 0.6) + ((1 - transport_score_flat) * 0.4)

# Scale to 0-100 for the final output
final_score_100 = final_score_flat * 100

# Add to GDF for export
gdf_grid_joined['Opportunity_Score'] = final_score_100

# --- Step 4: Visualization (Opportunity Map) ---
# Reshape final score back to grid shape for plotting
final_score_grid = final_score_100.reshape(xx.shape)

fig3, ax3 = plt.subplots(figsize=(20, 20))

# Plot Base Map
gdf_madrid_boundaries.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)

# Plot Opportunity Score as a Heatmap/Grid
# User wants Gradient: Red (High) - Blue (Low).
# 'coolwarm' goes Blue(low) -> Red(high).
cax3 = ax3.imshow(
    np.rot90(final_score_grid), 
    cmap='coolwarm', 
    extent=[xmin, xmax, ymin, ymax],
    alpha=0.6, 
    zorder=2, 
    aspect='auto'
)

ax3.set_title("Madrid Market Opportunity Map (Red=High, Blue=Low)", fontsize=20)
ax3.axis('off')

cbar3 = fig3.colorbar(cax3, ax=ax3, fraction=0.03, pad=0.04)
cbar3.set_label('Opportunity Score (0-100)')

plt.savefig("OpportunityMap.png", dpi=300, bbox_inches='tight')
print("Image saved as 'OpportunityMap.png'")


# --- Step 5: Export Ranking to Excel ---
# Create DataFrame for export
# We need Lat/Lon in degrees (EPSG:4326) for the report
gdf_grid_joined_wgs84 = gdf_grid_joined.to_crs(epsg=4326)

df_export = pd.DataFrame({
    'Ranking': range(1, len(gdf_grid_joined) + 1), # Placeholder, will sort first
    'Lat': gdf_grid_joined_wgs84.geometry.y,
    'Lon': gdf_grid_joined_wgs84.geometry.x,
    'Municipio': gdf_grid_joined['DS_NOMBRE'],
    'Score': gdf_grid_joined['Opportunity_Score']
})

# Sort by Score Descending
df_export = df_export.sort_values(by='Score', ascending=False).reset_index(drop=True)
df_export['Ranking'] = df_export.index + 1

# Select top 1000 or all? User showed a list. Let's save all or top 10k.
# 500x500 grid over Madrid is ~10k points. Saving all is fine.
output_csv = "OpportunityRanking.csv"
df_export.to_csv(output_csv, index=False, sep=';', encoding='utf-8-sig') # Using ; for Excel compatibility in Europe
print(f"Ranking saved to '{output_csv}'")

print("Top 5 Opportunity Zones:")
print(df_export.head())
