# Final Project Report: Madrid Market Opportunity & Transport Gap Analysis

### DATA COLLECTION & VISUALIZATION

> **Author:** Juan de Frutos  
> **Date:** 8/12/2025

---

## 1. Abstract

This project aims to identify and rank high-potential neighborhoods/areas (**"hot zones"**) in Madrid for the successful acquisition of the first clients for a transportation software system. The objective is to pinpoint areas with the highest concentration of the target market (students aged 18-25) combined with the poorest access to public transportation.

By synthesizing data from the Regional Transport Consortium (CRTM) and the National Statistics Institute (INE), the project utilizes **geospatial visualization** and **suitability modeling** to solve the "critical mass" problem for early adoption. Furthermore, the project implements a **Machine Learning "Gap Analysis"** to quantify infrastructure deficits, training a regression model to predict where transport nodes *should* exist based on population density and surrounding connectivity.

---

## 2. Data Collection

The project synthesizes data from three primary domains, fulfilling the requirement of over 1,000 items and multiple attributes:

### 1. Public Transport Infrastructure (CRTM)
* **Source:** `datos.crtm.es` (Open Data Portal).
* **Method:** Downloaded as CSV files.
* **Content:** Precise geographic coordinates (latitude, longitude) for five distinct modes of transport: **EMT Buses**, **Interurban Buses**, **Urban Buses**, **Metro Stations**, and **Cercanías** (Commuter Rail). These raw datasets contain over 10,000 individual stops each.

### 2. Population Demographics (INE)
* **Source:** `ine.es` (Spanish National Statistics Institute).
* **Method:** Downloaded full dataset, and then filtered by age group (18-25 years), sex, and municipality/district.
* **Content:** +1,000,000 rows with age, year, and municipality combinations.

### 3. Administrative Boundaries (GeoPackage)
* **Source:** `idem.comunidad.madrid/catalogocartografia/`
* **Content:** Polygons representing the 181 municipalities of Madrid. Used to spatially join population data to geographic coordinates.
* **Metadata:** Coordinate System ETRS89/ UTM zone 30N (converted to match transport data).

---

## 3. Visualization & Code Explanation

The following visual outputs were generated to explore the data and present the Opportunity Score.

### A. Student Density Map (Choropleth)
* **Visual:** `ChoroplethMap.png`. Colors each municipality based on student population density.

> **💡 Code Explanation:** > With `list_population_municipality`, once filled and filtered for the 18-25 age group using the most recent data (2022), we perform a merge. We match the municipality code from `list_population_municipality` with `gdf_madrid_boundaries` to create a common table. This ensures every geometry polygon has an associated population. We perform a **Left Join** (keeping the columns from `gdf_madrid_boundaries`) and assign the name `DS_NOMBRE_pop` to the right table to differentiate them.

### B. Transport Access Map (Heatmap/KDE)
* **Visual:** `HeatMap.png`. Uses Kernel Density Estimation to show transport concentration.

> **💡 Code Explanation (Kernel Logic & Math):** > The KDE algorithm aggregates the influence of individual transport stops to estimate a continuous density surface across the Madrid grid.
>
> * **The Multivariate Normal Distribution:** Instead of treating each stop as a single point, we model it as a Multivariate Normal Distribution. This creates a 3D "bell surface" defined by a random vector $X \in R^2$ (composed of Longitude and Latitude). The height of this surface, $f(x)$, represents the density intensity at any given coordinate.
> * **Quadratic Form & Covariance:** To calculate the bell shape mathematically, the function relies on the exponent of a quadratic form. The algorithm calculates the transpose of the mean-centered vector multiplied by the inverse of the Covariance Matrix.
> * **The Summation (Algorithm):** The final density value for any specific point $(x, y)$ on our mesh grid is the summation of the density contributions from all surrounding transport stops.
> * **Normalization:** As the value of each gap is really small because the probability density needs to be 1 in total, we normalize it, and we multiply it by 100 so it is ranked from 0-100 (`f_grid` variable).

### C. Opportunity Map Model (The Solution)
* **Visual:** `OpportunityMap.png`. A Grid Map where each cell is colored by the final **"Opportunity Score"**.

### D. TOTAL_GAP (Machine Learning)
* **Visual:** `Total_Gap_Map.png`. Displays the Total Infrastructure Gap (Deficit Accumulation).

---

## 4. Modeling & Prediction

### Model A: Heuristic Opportunity Score
* **Type:** Suitability Analysis (Weighted Scoring).
* **Method:** `Final_Score = (Pop_Score * 0.6) + ((1 - Transport_Score) * 0.4)`
* **Variables:** Student population density (Positive impact) and transport access index (Negative impact).

> **💡 Code Explanation (Grid & Scoring):** > `gdf_grid` is a geodataframe of points, which basically comes from `x_flat = xx.ravel()` and `y_flat = yy.ravel()` created in the KDE step. These are arrays of points into which we have divided the map. These points are in `epsg=3857` coordinates so they match our geodataframe data for stops and municipality boundaries.
>
> * **How mgrid works:** `numpy.mgrid` and the matrices `xx`, `yy` do NOT store boxes or areas. They store points (vertices).
> * **The Join:** `gdf_grid_joined = gpd.sjoin(gdf_grid, gdf_choropleth[['geometry', 'DS_NOMBRE', 'Pop_Score']], how='left', predicate='within')`. Here, the code goes point by point through the grid and checks if it falls within the municipality geometry. If yes, it assigns the municipality name and score to that specific point row. We use `fillna(0)` because some points may not belong to any municipality.

### Model B: Machine Learning (Gap Analysis)
This section replaces standard classification/regression with a custom **"Gap Analysis"** to predict infrastructure deficits.

#### Grid Generation and Variable Assignment
* **Code:** `cols = np.arange(xmin, xmax, 500)`
* **Shape:** PERFECT SQUARE (500m x 500m) = 0.25 km².
* **Consequence:** You cover 100% of Madrid. What "remains" simply falls into "no man's land"; the `sjoin` will say there is no municipality there and assign population 0.
* **Feature Engineering:** A loop creates an array of polygons (squares). We verify how many stops are inside each polygon using `sjoin`. We also generate `gdf_ml_grid['centroid']` to assign Population Density to each grid square using a `within` join with the population choropleth.

We compared two distinct ML approaches:

#### 🧠 ML Model 2: Iterative Multimodal Analysis (Preferred)
* **Dataset:** Synthetic `TOTAL_GAP` dataset.
* **Features (X):**
    * **Local:** `Pop_Local`, `Count_{Mode}` (5 independent columns: Interurban, Urban, Metro, EMT, Light Rail).
    * **Context (Neighbors):** `Stops_3x3` (Sum of all adjacent stops), `Stops_5x5` (Sum of all surrounding stops).
* **Algorithm:** Round Robin (Leave-One-Out) using **Random Forest Regressor**.
* **Process:** Loops 5 times. In each iteration, one transport mode is the Target (Y), and the others + Population are predictors (X).
* **Gap Calculation:** `Deficit = Prediction_Ideal - Reality`. If Deficit > 0, it is stored.
* **Result:** Summation of positive deficits.

#### 📉 ML Model 1: Aggregated Analysis
* **Features (X):**
    * **Local:** `Pop_Local`, `Count_{Total}` (1 column summing all stops per polygon regardless of mode).
    * **Context:** `Stops_3x3`, `Stops_5x5`.
* **Difference:** Treats transport as a monolith rather than distinct modes.

#### ⚙️ ML Model 3: Balanced Baseline Configuration
* **Parameters:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`.
* **Technical Rationale:** This setup serves as the high-variance baseline. With a standard learning rate (0.1) and deeper trees (5), the model can rapidly capture complex, non-linear interactions between population density and transport nodes.

#### 🐢 ML Model 4: Robust "Slow Learner" Configuration
* **Parameters:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`.
* **Technical Rationale:** Implements a more aggressive regularization strategy.
    * **Shrinkage:** Halving the learning rate to 0.05 forces the model to learn more gradually.
    * **Compensation:** Doubling estimators (200) ensures convergence.
    * **Complexity Control:** Reducing tree depth to 4 limits complexity, forcing reliance on the collective vote rather than specific features.
    * **Implication:** Designed for generalization.

---

## 5. Results & Conclusions

* **Heuristic Model (Opportunity Score):** Successfully generated a ranked list (CSV) of "Hot Zones" for business logic. It identified high-density student areas like **Leganés (Norte)** and **San Blas**, providing immediate actionable coordinates for the launch.

* **Machine Learning Comparison (Gap Analysis):**
    * **Strategy 1 (Aggregated - Model 1):** Produced a high-variance map with extreme deficit values. **Flaw:** By treating a Metro station and a Bus stop as mathematically equivalent, it failed to capture specific utility.
    * **Strategy 2 (Granular - Model 2):** Produced a more refined map with lower, more realistic deficit values. **Benefit:** The algorithm successfully learned the relationships between modes (e.g., intermodal complementarity), reducing "false positive" gaps.

**Final Conclusion:** Model 2 (Granular/Multiple Columns) is the superior model for urban planning. It demonstrates that when we account for the specific type of transport available, the actual gap prediction becomes significantly more accurate.
