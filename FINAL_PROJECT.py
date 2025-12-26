
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from shapely.geometry import box
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import random


# --- CONFIGURACION ---

PLOT_BACKGROUND_COLOR = '#D9D9D9' 
PLOT_CMAP = 'turbo'

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
        print("::::::::::::::::::::::::::::::::::::::::::" + "\n")
        print("GDF_MADRID_BOUNDARIES BEFORE TRANSFORMATION :", gdf_madrid_boundaries.head())
        gdf_madrid_boundaries = gdf_madrid_boundaries.to_crs(epsg=3857) # transformation to epsg 3857 standard for all of my maps.
        # as idem_cm_unid_admin is in EPSG:25830 (ETRS89 UTM uso 30)
        print("GDF_MADRID_BOUNDARIES AFTER TRANSFORMATION :", gdf_madrid_boundaries.head())
        print("GDF_MADRID_BOUNDARIES INFO :", gdf_madrid_boundaries.info())
        print("::::::::::::::::::::::::::::::::::::::::::" + "\n")

        xmin, ymin, xmax, ymax = gdf_madrid_boundaries.total_bounds
        width_m = xmax - xmin
        height_m = ymax - ymin

        print("\n" + "=" * 60)
        print(f"  DIMENSIONES TOTALES DEL MAPA DE MADRID (Bounding Box)")
        print("=" * 60)
        print(f" X Min (Oeste): {xmin:,.2f}")
        print(f" X Max (Este) : {xmax:,.2f}")
        print(f" Y Min (Sur)  : {ymin:,.2f}")
        print(f" Y Max (Norte): {ymax:,.2f}")
        print("-" * 60)
        print(f" ANCHO TOTAL : {width_m / 1000:,.2f} km ({width_m:,.0f} m)")
        print(f" ALTO TOTAL  : {height_m / 1000:,.2f} km ({height_m:,.0f} m)")
        print(f" ÁREA RECTANGULAR: {(width_m * height_m) / 1e6:,.2f} km²")
        print("=" * 60 + "\n")


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
                print(":::::::::::::::::::::::::::::::::::::::::: " + "\n")
                print("GDF_STOPS HEAD :" , str(gdf_stops.head()) , "\n type :" , str(type(gdf_stops)) , "\n geometry :" , str(gdf_stops.geometry) , "\n geometry.x :" , str(gdf_stops.geometry.x))
                print("GDF_STOPS HEAD WITHOUT STR :" , gdf_stops.head() , "\n type :" , type(gdf_stops) , "\n geometry :" , gdf_stops.geometry  , "\n geometry.x :" , gdf_stops.geometry.x)
                print(":::::::::::::::::::::::::::::::::::::::::: " + "\n")

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
        df_filtered = df[mask_sexo & mask_fecha].copy()# Without .copy(), df_filtered is the same "sheet of paper" (original data), # you are just looking through a "window" (view).
        
        # Format is like "0 años", "1 año", "100 años ", "need to filter it "
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
        print("\n--- TO VISUALICE IF THERE IS NAN VALUES ---")
        print(df_final.head(10))  #
        print("Tipos de datos:\n", df_final.dtypes)
        print("--------------------------------------------------\n")
        # Clean Total column (remove dots if present for thousands separator)
        df_final["Total"] = df_final["Total"].astype(str).str.replace('.', '', regex=False)
        df_final["Total"] = pd.to_numeric(df_final["Total"], errors='coerce')
        
        df_final = df_final.groupby(['CD_INE', 'DS_NOMBRE'])['Total'].sum().reset_index()
        list_pupulation_municipality = df_final

        print(":::::::::::::::::::::::::::::::::::::::::: " + "\n")
        print("LIST_PUPULATION_MUNICIPALITY INFO :", list_pupulation_municipality.info())
        print("LIST_PUPULATION_MUNICIPALITY HEAD :", list_pupulation_municipality.head())
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::: " + "\n")

    except Exception as e:
        print(f"Error adding {ARCHIVO_POB}: {e}")
        import traceback
        traceback.print_exc()
        exit()
    

    # ::::::::::::::::::::::: MACHINE LEARNING MODEL 1 :::::::::::::::::::::::::

# ::::::::::::::::::::::: MACHINE LEARNING HELPERS :::::::::::::::::::::::::

def create_ml_grid(xmin, xmax, ymin, ymax, gdf_choropleth):
    CELL_SIZE = 500
    cols = np.arange(xmin, xmax, CELL_SIZE)
    rows = np.arange(ymin, ymax, CELL_SIZE)

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(box(x, y, x + CELL_SIZE, y + CELL_SIZE))

    gdf_ml_grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:3857")
    print(f"Created ML Grid with {len(gdf_ml_grid)} cells.")





    sample_area_m2 = gdf_ml_grid.geometry.iloc[0].area

    print("\n" + "#" * 60)
    print(f"  DETALLES DE LA MALLA (GRID) GENERADA")
    print("#" * 60)
    print(f"  Tamaño Celda Configurado: {CELL_SIZE} x {CELL_SIZE} metros")
    print(f"  Área Calculada por Celda: {sample_area_m2:,.0f} m² ({sample_area_m2 / 1e6:.4f} km²)")
    print(f"  Resolución de la Malla  : {len(cols)} columnas x {len(rows)} filas")
    print(f"  Total de Celdas (Polígonos): {len(gdf_ml_grid):,}")
    print("#" * 60 + "\n")





    gdf_ml_grid['centroid'] = gdf_ml_grid.geometry.centroid
    
    gdf_ml_grid_pop = gpd.sjoin(
        gpd.GeoDataFrame(gdf_ml_grid[['centroid']], geometry='centroid'),
        gdf_choropleth[['geometry', 'pop_density']], 
        how='left',
        predicate='within'
    )

    gdf_ml_grid['pop_density'] = gdf_ml_grid_pop['pop_density'].values
    gdf_ml_grid['pop_density'] = gdf_ml_grid['pop_density'].fillna(0)
    
    return gdf_ml_grid

def calculate_spatial_features(gdf_ml_grid, xmin, xmax, ymin, ymax):
    from scipy.ndimage import convolve
    
    CELL_SIZE = 500
    cols_count = int((xmax - xmin) / CELL_SIZE) 
    rows_count = int((ymax - ymin) / CELL_SIZE)
    
    
    unique_x = sorted(gdf_ml_grid.geometry.centroid.x.unique())
    unique_y = sorted(gdf_ml_grid.geometry.centroid.y.unique())
    shape = (len(unique_x), len(unique_y))
    
    mode_cols = [c for c in gdf_ml_grid.columns if c.startswith('count_')]
    total_stops_series = gdf_ml_grid[mode_cols].sum(axis=1)
    
    grid_matrix = total_stops_series.values.reshape(shape)
    
    k3 = np.ones((3, 3))
    k3[1, 1] = 0
    
    k5 = np.ones((5, 5))
    k5[2, 2] = 0 
    
    
    stops_3x3_grid = convolve(grid_matrix, k3, mode='constant', cval=0)
    
    stops_5x5_grid = convolve(grid_matrix, k5, mode='constant', cval=0)
    
    gdf_ml_grid['Stops_3x3'] = stops_3x3_grid.ravel()
    gdf_ml_grid['Stops_5x5'] = stops_5x5_grid.ravel()
    
    return gdf_ml_grid

def calculate_all_stop_counts(gdf_ml_grid, list_dfs_stops):
    
    def count_points_in_polygons(polygons_gdf, points_gdf):
        if points_gdf is None or points_gdf.empty:
            return np.zeros(len(polygons_gdf))
        
        points_gdf = points_gdf.copy()
        points_gdf['count_val'] = 1

        joined = gpd.sjoin(points_gdf, polygons_gdf[['geometry']], how='inner', predicate='within')
        counts = joined.groupby('index_right')['count_val'].count()
        
        return polygons_gdf.index.map(counts).fillna(0).astype(int)

    print("Calculating features (stop counts per cell) for ALL modes...")
    
    MODE_NAMES = ["Interurban", "Urban", "Metro", "EMT", "Metro_Ligero"]
 
    for i, name in enumerate(MODE_NAMES):
        col_name = f'count_{name}'
        print(f"  Counting stops for {name}...")
        gdf_ml_grid[col_name] = count_points_in_polygons(gdf_ml_grid, list_dfs_stops[i])
        
    return gdf_ml_grid, MODE_NAMES


# ::::::::::::::::::::::: MACHINE LEARNING MODEL 1 :::::::::::::::::::::::::

def ml_model(gdf_ml_grid, MODE_NAMES, gdf_madrid_boundaries):

    print("\n::::::::::::::::::::::: RUNNING MODEL 1 (Gap Analysis, single stops columns) :::::::::::::::::::::::")
    
    gdf_ml_grid['TOTAL_GAP'] = 0.0
    deficit_cols = []

    for i, target_name in enumerate(MODE_NAMES):

        print(f"\n--- Training Model 1 for Target: {target_name} ---")
        target_col = f'count_{target_name}'
        y = gdf_ml_grid[target_col]
        
        other_modes = [name for j, name in enumerate(MODE_NAMES) if j != i]
        other_cols = [f'count_{name}' for name in other_modes]
        
        gdf_ml_grid['count_total_stops'] = gdf_ml_grid[other_cols].sum(axis=1)
        
        feature_cols = ['pop_density', 'Stops_3x3', 'Stops_5x5', 'count_total_stops']
        
        X = gdf_ml_grid[feature_cols]
        
        print(f"Features for {target_name}: {feature_cols}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)
        
    
        y_pred_test = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        print(f"R2 Score: {r2:.4f}")
        
       
        pred_col = f'pred_ideal_{target_name}'
        gdf_ml_grid[pred_col] = rf_model.predict(X)
        deficit_col = f'deficit_{target_name}'
        gdf_ml_grid[deficit_col] = gdf_ml_grid[pred_col] - gdf_ml_grid[target_col]
        gdf_ml_grid[deficit_col] = gdf_ml_grid[deficit_col].apply(lambda x: x if x > 0 else 0)
        
        deficit_cols.append(deficit_col)
        gdf_ml_grid['TOTAL_GAP'] += gdf_ml_grid[deficit_col]
        
        fig_ml, (ax_ml1, ax_ml2) = plt.subplots(1, 2, figsize=(20, 10), facecolor=PLOT_BACKGROUND_COLOR)
        ax_ml1.set_facecolor(PLOT_BACKGROUND_COLOR)
        ax_ml2.set_facecolor(PLOT_BACKGROUND_COLOR)
        
        gdf_madrid_boundaries.plot(ax=ax_ml1, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
        gdf_ml_grid.plot(column=target_col, ax=ax_ml1, cmap=PLOT_CMAP, legend=True, alpha=0.6)
        ax_ml1.set_title(f"Actual {target_name} Density")
        ax_ml1.axis('off')
        
        gdf_madrid_boundaries.plot(ax=ax_ml2, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
        gdf_ml_grid.plot(column=pred_col, ax=ax_ml2, cmap=PLOT_CMAP, legend=True, alpha=0.6)
        ax_ml2.set_title(f"Model 2 Predicted {target_name} (Ideal) (R2={r2:.2f})")
        ax_ml2.axis('off')
        
        output_img = f"ML1_GapAnalysis_single_stop_column_{target_name}.png"
        
        
        fig_ml.text(0.5, 0.02, "Color Scale: Stops per 500x500m cell (Blue=0, Yellow=Max)", ha='center', fontsize=12)
        
        plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
        print(f"Saved plot: {output_img}")
        plt.close(fig_ml)


    print("\nGenerating TOTAL GAP Map...")
    fig_gap, ax_gap = plt.subplots(figsize=(12, 12), facecolor=PLOT_BACKGROUND_COLOR)
    ax_gap.set_facecolor(PLOT_BACKGROUND_COLOR)
    
    gdf_madrid_boundaries.plot(ax=ax_gap, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
    # Plot only non-zero gap with a small threshold to avoid minimal noise
    gdf_plot_gap = gdf_ml_grid[gdf_ml_grid['TOTAL_GAP'] > 0.05]
    if not gdf_plot_gap.empty:
        gdf_plot_gap.plot(column='TOTAL_GAP', ax=ax_gap, cmap=PLOT_CMAP, legend=True, alpha=0.6)
    ax_gap.set_title("TOTAL INFRASTRUCTURE GAP (Deficit Accumulation)", fontsize=15)
    ax_gap.axis('off')
    
    plt.savefig("TOTAL_GAP_Map_single_stop_column.png", dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
    print("Saved 'TOTAL_GAP_Map_single_stop_column.png'")
    

    output_csv = "GAP_Analysis_Results_single_stop_column.csv"
    
    export_cols = ['geometry', 'pop_density', 'TOTAL_GAP'] + deficit_cols
    gdf_export = gdf_ml_grid[export_cols].copy()
    gdf_export['lat'] = gdf_export.geometry.centroid.to_crs(epsg=4326).y
    gdf_export['lon'] = gdf_export.geometry.centroid.to_crs(epsg=4326).x
    
    df_export = pd.DataFrame(gdf_export.drop(columns=['geometry']))
    df_export.to_csv(output_csv, index=False, sep=';')
    print(f"Saved GAP analysis data to '{output_csv}'")


# ::::::::::::::::::::::: MACHINE LEARNING MODEL 2 :::::::::::::::::::::::::
def ml_model_2(gdf_ml_grid, MODE_NAMES, gdf_madrid_boundaries):
    
    print("\n::::::::::::::::::::::: RUNNING MODEL 2 (Gap Analysis, multiple stops columns) :::::::::::::::::::::::")
    
    gdf_ml_grid['TOTAL_GAP'] = 0.0
    deficit_cols = []

    for i, target_name in enumerate(MODE_NAMES):
        print(f"\n--- ----- Iteration {i+1}/5: Target = {target_name} ---------")
        target_col = f'count_{target_name}'
        
        other_modes = [name for j, name in enumerate(MODE_NAMES) if j != i]
        other_cols = [f'count_{name}' for name in other_modes]
        
        feature_cols = ['pop_density', 'Stops_3x3', 'Stops_5x5'] + other_cols
        
        X = gdf_ml_grid[feature_cols]
        y = gdf_ml_grid[target_col]
        
        print(f"Features: {feature_cols}")
        
      
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)
        
    
        y_pred_test = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        print(f"R2 Score: {r2:.4f}")
        
       
        pred_col = f'pred_ideal_{target_name}'
        gdf_ml_grid[pred_col] = rf_model.predict(X)
        deficit_col = f'deficit_{target_name}'
        gdf_ml_grid[deficit_col] = gdf_ml_grid[pred_col] - gdf_ml_grid[target_col]
        gdf_ml_grid[deficit_col] = gdf_ml_grid[deficit_col].apply(lambda x: x if x > 0 else 0)
        deficit_cols.append(deficit_col)
        gdf_ml_grid['TOTAL_GAP'] += gdf_ml_grid[deficit_col]
        fig_ml, (ax_ml1, ax_ml2) = plt.subplots(1, 2, figsize=(20, 10), facecolor=PLOT_BACKGROUND_COLOR)
        ax_ml1.set_facecolor(PLOT_BACKGROUND_COLOR)
        ax_ml2.set_facecolor(PLOT_BACKGROUND_COLOR)
        
        gdf_madrid_boundaries.plot(ax=ax_ml1, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
        gdf_ml_grid.plot(column=target_col, ax=ax_ml1, cmap=PLOT_CMAP, legend=True, alpha=0.6)

        ax_ml1.set_title(f"Actual {target_name} Density")
        ax_ml1.axis('off')
        
        gdf_madrid_boundaries.plot(ax=ax_ml2, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
        gdf_ml_grid.plot(column=pred_col, ax=ax_ml2, cmap=PLOT_CMAP, legend=True, alpha=0.6)
        ax_ml2.set_title(f"Model 2 Predicted {target_name} (Ideal) (R2={r2:.2f})")
        ax_ml2.axis('off')
        
        output_img = f"ML2_GapAnalysis_multiple_stop_columns_{target_name}.png"
        
       
        fig_ml.text(0.5, 0.02, "Color Scale: Stops per 500x500m cell (Blue=0, Yellow=Max)", ha='center', fontsize=12)
        
        plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
        print(f"Saved plot: {output_img}")
        plt.close(fig_ml)


    print("\nGenerating TOTAL GAP Map...")
    fig_gap, ax_gap = plt.subplots(figsize=(12, 12), facecolor=PLOT_BACKGROUND_COLOR)
    ax_gap.set_facecolor(PLOT_BACKGROUND_COLOR)
    
    gdf_madrid_boundaries.plot(ax=ax_gap, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
    gdf_plot_gap = gdf_ml_grid[gdf_ml_grid['TOTAL_GAP'] > 0.05]
    if not gdf_plot_gap.empty:
        gdf_plot_gap.plot(column='TOTAL_GAP', ax=ax_gap, cmap=PLOT_CMAP, legend=True, alpha=0.6)
    ax_gap.set_title("TOTAL INFRASTRUCTURE GAP (Deficit Accumulation)", fontsize=15)
    ax_gap.axis('off')
    
    plt.savefig("TOTAL_GAP_Map_multiple_stop_columns.png", dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
    print("Saved 'TOTAL_GAP_Map_multiple_stop_columns.png'")
    

    output_csv = "GAP_Analysis_Results_multiple_stop_columns.csv"
    
    export_cols = ['geometry', 'pop_density', 'TOTAL_GAP'] + deficit_cols
    gdf_export = gdf_ml_grid[export_cols].copy()
    gdf_export['lat'] = gdf_export.geometry.centroid.to_crs(epsg=4326).y
    gdf_export['lon'] = gdf_export.geometry.centroid.to_crs(epsg=4326).x
    
    df_export = pd.DataFrame(gdf_export.drop(columns=['geometry']))
    df_export.to_csv(output_csv, index=False, sep=';')
    print(f"Saved GAP analysis data to '{output_csv}'")


# ::::::::::::::::::::::: MACHINE LEARNING MODEL 3 :::::::::::::::::::::::::
def ml_model_3(gdf_ml_grid, MODE_NAMES, gdf_madrid_boundaries):
    
    print("\n::::::::::::::::::::::: RUNNING MODEL 3 (GradientBoostingRegressor 1) :::::::::::::::::::::::")
    
    for i, target_name in enumerate(MODE_NAMES):
        print(f"\n--- Model 3 - Iteration {i+1}/5: Target = {target_name} ---")
        target_col = f'count_{target_name}'
        
        other_modes = [name for j, name in enumerate(MODE_NAMES) if j != i]
        other_cols = [f'count_{name}' for name in other_modes]
        
        feature_cols = ['pop_density', 'Stops_3x3', 'Stops_5x5'] + other_cols
        
        X = gdf_ml_grid[feature_cols]
        y = gdf_ml_grid[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        gb_model.fit(X_train, y_train)
        
        y_pred_test = gb_model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        print(f"R2 Score: {r2:.4f}")
        
        pred_col = f'pred_ideal_{target_name}_m3'
        gdf_ml_grid[pred_col] = gb_model.predict(X)
    
        
        fig_ml, (ax_ml1, ax_ml2) = plt.subplots(1, 2, figsize=(20, 10), facecolor=PLOT_BACKGROUND_COLOR)
        ax_ml1.set_facecolor(PLOT_BACKGROUND_COLOR)
        ax_ml2.set_facecolor(PLOT_BACKGROUND_COLOR)
        
        gdf_madrid_boundaries.plot(ax=ax_ml1, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
        gdf_ml_grid.plot(column=target_col, ax=ax_ml1, cmap=PLOT_CMAP, legend=True, alpha=0.6)
        ax_ml1.set_title(f"Actual {target_name} Density")
        ax_ml1.axis('off')
        
        gdf_madrid_boundaries.plot(ax=ax_ml2, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
        gdf_ml_grid.plot(column=pred_col, ax=ax_ml2, cmap=PLOT_CMAP, legend=True, alpha=0.6)
        ax_ml2.set_title(f"Model 3 Predicted {target_name} (R2={r2:.2f})")
        ax_ml2.axis('off')
        
        output_img = f"ML3_GapAnalysis_GradientBoostingRegressor_{target_name}.png"
        
        fig_ml.text(0.5, 0.02, "Color Scale: Stops per cell", ha='center', fontsize=12)
        
        plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
        print(f"Saved plot: {output_img}")
        plt.close(fig_ml)


# ::::::::::::::::::::::: MACHINE LEARNING MODEL 4 :::::::::::::::::::::::::
def ml_model_4(gdf_ml_grid, MODE_NAMES, gdf_madrid_boundaries):
    
    print("\n::::::::::::::::::::::: RUNNING MODEL 4 (GradientBoostingRegressor 2) :::::::::::::::::::::::")
    
    for i, target_name in enumerate(MODE_NAMES):
        print(f"\n--- Model 4 - Iteration {i+1}/5: Target = {target_name} ---")
        target_col = f'count_{target_name}'
        
        other_modes = [name for j, name in enumerate(MODE_NAMES) if j != i]
        other_cols = [f'count_{name}' for name in other_modes]
        
        feature_cols = ['pop_density', 'Stops_3x3', 'Stops_5x5'] + other_cols
        
        X = gdf_ml_grid[feature_cols]
        y = gdf_ml_grid[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # changed parameters for Model 4 ---- Different from Model 3
        gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        gb_model.fit(X_train, y_train)
        
        y_pred_test = gb_model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        print(f"R2 Score: {r2:.4f}")
        
        pred_col = f'pred_ideal_{target_name}_m4'
        gdf_ml_grid[pred_col] = gb_model.predict(X)
        
        
        fig_ml, (ax_ml1, ax_ml2) = plt.subplots(1, 2, figsize=(20, 10), facecolor=PLOT_BACKGROUND_COLOR)
        ax_ml1.set_facecolor(PLOT_BACKGROUND_COLOR)
        ax_ml2.set_facecolor(PLOT_BACKGROUND_COLOR)
        
        gdf_madrid_boundaries.plot(ax=ax_ml1, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
        gdf_ml_grid.plot(column=target_col, ax=ax_ml1, cmap=PLOT_CMAP, legend=True, alpha=0.6)
        ax_ml1.set_title(f"Actual {target_name} Density")
        ax_ml1.axis('off')
        
        gdf_madrid_boundaries.plot(ax=ax_ml2, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)
        gdf_ml_grid.plot(column=pred_col, ax=ax_ml2, cmap=PLOT_CMAP, legend=True, alpha=0.6)
        ax_ml2.set_title(f"Model 4 Predicted {target_name} (R2={r2:.2f})")
        ax_ml2.axis('off')
        
        output_img = f"ML4_GapAnalysis_GradientBoostingRegressor_{target_name}.png"
        
        fig_ml.text(0.5, 0.02, "Color Scale: Stops per cell", ha='center', fontsize=12)
        
        plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
        print(f"Saved plot: {output_img}")
        plt.close(fig_ml)
        
 

# ::::::::::::::::::::::: EXECUTION FOR THE MODELS :::::::::::::::::::::::::


if __name__ == "__main__":
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

    print(":::::::::::::::::::::::::::::::::::::::::::::::::::: " + "\n")
    print("GDF_CHOROPLETH INFO :", gdf_choropleth.iloc[0])
    print("GDF_CHOROPLETH HEAD :", gdf_choropleth.head())
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::: " + "\n")


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
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::: " + "\n")
    print("VALUES OF VALUES IN KDE :", values[1,random.randint(0, len(values)-1)], type(values), len(values))
    print("POSITIONS OF POSITIONS IN KDE :", positions[1,random.randint(0, len(positions)-1)], type(positions), len(positions))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::: " + "\n")
    # KDE
    kernel = gaussian_kde(values, bw_method=0.05)#resultand function: sum of the all the gauss bells of all the stops

    f = kernel(positions)

    f_norm = (f - f.min()) / (f.max() - f.min()) * 100  #as we have density prob expressions: 3.4e-05 we nned to normalize
    f_log = np.log1p(f)
    f_norm_log = (f_log - f_log.min()) / (f_log.max() - f_log.min()) * 100 #logaritmical normalization
    f_grid = f_norm_log.reshape(xx.shape)
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::: " + "\n")
    print("VIS OF F_NORM :", f_norm.max(), type(f_norm))
    print("VIS OF F :", f.max(), type(f))
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::: " + "\n")


    #------ Generating the plot ----

    fig, ax = plt.subplots(figsize=(12, 12), facecolor=PLOT_BACKGROUND_COLOR)
    ax.set_facecolor(PLOT_BACKGROUND_COLOR)

    gdf_madrid_boundaries.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)


    cax = ax.imshow(np.rot90(f_grid), cmap=PLOT_CMAP, extent=[xmin, xmax, ymin, ymax],
                    alpha=0.5, zorder=2, aspect='auto') 

    ax.set_title("Transport Density in Madrid + base map", fontsize=15)

    cbar = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Transport Density (0-100)')


    plt.savefig("HeatMap.png", dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
    print("Image saved as 'HeatMap.png'")


    # ::::::::::::::::::::::: CHOROPLETH MAP :::::::::::::::::::::::::


    fig2, ax2 = plt.subplots(figsize=(20, 20), facecolor=PLOT_BACKGROUND_COLOR)
    ax2.set_facecolor(PLOT_BACKGROUND_COLOR)
    gdf_madrid_boundaries.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=0.3, alpha=0.5)

    plot_choro = gdf_choropleth.plot(
        ax=ax2,
        column='Total',
        cmap=PLOT_CMAP,      
        legend=True,
        legend_kwds={'label': "Población Joven (18-24 años)", 'shrink': 0.6},
        edgecolor='white',
        linewidth=0.3
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

    plt.savefig("ChoroplethMap.png", dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
    print("Image saved as 'ChoroplethMap.png'")

    # ::::::::::::::::::::::: OPPORTUNITY SCORE MODEL :::::::::::::::::::::::::

    # Ensure CRS is projected (meters) for area calculation. It is already EPSG:3857.
    gdf_choropleth['area_km2'] = gdf_choropleth.geometry.area / 10**6
    gdf_choropleth['pop_density'] = gdf_choropleth['Total'] / gdf_choropleth['area_km2']


    gdf_choropleth['pop_density_log'] = np.log1p(gdf_choropleth['pop_density']) # Normalize Population Density (Log Normalization 0-1)
    # Add a small constant to avoid log(0)
    min_pop = gdf_choropleth['pop_density_log'].min()
    max_pop = gdf_choropleth['pop_density_log'].max()
    gdf_choropleth['Pop_Score'] = (gdf_choropleth['pop_density_log'] - min_pop) / (max_pop - min_pop)
    gdf_choropleth['Pop_Score'] = gdf_choropleth['Pop_Score'].fillna(0)

    x_flat = xx.ravel() # We reuse xx, yy from the KDE step.
    y_flat = yy.ravel()

    # Create a GeoDataFrame for the grid points
    gdf_grid = gpd.GeoDataFrame(
        {'geometry': gpd.points_from_xy(x_flat, y_flat)}, #transforms the array of points into a geometry object
        crs="EPSG:3857"
    )

    gdf_grid_joined = gpd.sjoin( #assign Municipality info to each grid point
        gdf_grid, 
        gdf_choropleth[['geometry', 'DS_NOMBRE', 'Pop_Score']], 
        how='left', 
        predicate='within'
    )
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("gdf_grid_joined info :", gdf_grid_joined.info())
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    gdf_grid_joined['Pop_Score'] = gdf_grid_joined['Pop_Score'].fillna(0)
    gdf_grid_joined['DS_NOMBRE'] = gdf_grid_joined['DS_NOMBRE'].fillna("Unknown")



    #  f_grid comes from f_norm_log which is 0-100. So we need Transport_Score in 0-1 range.
    transport_score_flat = f_grid.ravel() / 100.0 
    pop_score_flat = gdf_grid_joined['Pop_Score'].values
    final_score_flat = (pop_score_flat * 0.6) + ((1 - transport_score_flat) * 0.4)
    final_score_100 = final_score_flat * 100
    gdf_grid_joined['Opportunity_Score'] = final_score_100

    # Reshape final score back to grid shape for plotting
    final_score_grid = final_score_100.reshape(xx.shape)

    fig3, ax3 = plt.subplots(figsize=(20, 20), facecolor=PLOT_BACKGROUND_COLOR)
    ax3.set_facecolor(PLOT_BACKGROUND_COLOR)

    gdf_madrid_boundaries.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=0.3, alpha=0.5)

    # Plot Opportunity Score as a Heatmap/Grid
    # 'coolwarm' goes Blue(low) -> Red(high).
    cax3 = ax3.imshow(
        np.rot90(final_score_grid), 
        cmap=PLOT_CMAP, 
        extent=[xmin, xmax, ymin, ymax],
        alpha=0.6, 
        zorder=2, 
        aspect='auto'
    )

    ax3.set_title("Madrid Market Opportunity Map (Red=High, Blue=Low)", fontsize=20)
    ax3.axis('off')

    cbar3 = fig3.colorbar(cax3, ax=ax3, fraction=0.03, pad=0.04)
    cbar3.set_label('Opportunity Score (0-100)')

    plt.savefig("OpportunityMap.png", dpi=300, bbox_inches='tight', facecolor=PLOT_BACKGROUND_COLOR)
    print("Image saved as 'OpportunityMap.png'")



    # We need Lat/Lon in degrees (EPSG:4326) for the report
    gdf_grid_joined_wgs84 = gdf_grid_joined.to_crs(epsg=4326)

    df_export = pd.DataFrame({
        'Ranking': range(1, len(gdf_grid_joined) + 1),
        'Lat': gdf_grid_joined_wgs84.geometry.y,
        'Lon': gdf_grid_joined_wgs84.geometry.x,
        'Municipio': gdf_grid_joined['DS_NOMBRE'],
        'Score': gdf_grid_joined['Opportunity_Score']
    })

    # Sort by Score Descending
    df_export = df_export.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df_export['Ranking'] = df_export.index + 1

    output_csv = "OpportunityRanking.csv"
    df_export.to_csv(output_csv, index=False, sep=';', encoding='utf-8-sig') 
    print(f"Ranking saved to '{output_csv}'")

    print("Top 5 Opportunity Zones:")
    print(df_export.head())

    gdf_ml_grid = create_ml_grid(xmin, xmax, ymin, ymax, gdf_choropleth)
    gdf_ml_grid, MODE_NAMES = calculate_all_stop_counts(gdf_ml_grid, list_dfs_stops)

    gdf_ml_grid = calculate_spatial_features(gdf_ml_grid, gdf_ml_grid.total_bounds[0], gdf_ml_grid.total_bounds[2], gdf_ml_grid.total_bounds[1], gdf_ml_grid.total_bounds[3])

    ml_model(gdf_ml_grid, MODE_NAMES, gdf_madrid_boundaries)
    ml_model_2(gdf_ml_grid, MODE_NAMES, gdf_madrid_boundaries)
    ml_model_3(gdf_ml_grid, MODE_NAMES, gdf_madrid_boundaries)
    ml_model_4(gdf_ml_grid, MODE_NAMES, gdf_madrid_boundaries)
