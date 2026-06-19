# 🗺️ Geospatial Analysis & Mapping

> **Prerequisites**: Pandas, Data Visualization | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [What is Geospatial Data?](#1-what-is-geospatial-data)
2. [Coordinate Reference Systems (CRS)](#2-coordinate-reference-systems-crs)
3. [GeoPandas: Pandas for Maps](#3-geopandas-pandas-for-maps)
4. [Spatial Joins & Buffer Zones](#4-spatial-joins--buffer-zones)
5. [Interactive Mapping with Folium](#5-interactive-mapping-with-folium)
6. [Library Implementation (GeoPandas & Folium)](#6-library-implementation-geopandas--folium)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. What is Geospatial Data?

Data Science doesn't just happen in spreadsheets; it happens in the real world. Uber, Airbnb, and Zillow rely heavily on Geospatial Analysis to predict prices, route drivers, and match users.

Geospatial data comes in two main formats:
1. **Vector Data**: Discrete shapes.
   - **Points**: A specific location (e.g., A restaurant's Lat/Long).
   - **Lines**: A path (e.g., A road or a river).
   - **Polygons**: An enclosed area (e.g., A zip code, a lake, or a country border).
2. **Raster Data**: Continuous grids of pixels.
   - Usually satellite imagery or elevation maps, where each pixel holds a value (e.g., temperature).

---

## 2. Coordinate Reference Systems (CRS)

The Earth is a 3D sphere, but your computer screen is a 2D rectangle. 
Projecting a 3D sphere onto a 2D map mathematically guarantees that you will distort either **Area, Shape, Distance, or Direction**.

A **Coordinate Reference System (CRS)** defines exactly how the 3D coordinates (Latitude/Longitude) map to the 2D plane.

- **EPSG:4326 (WGS84)**: The global standard for GPS. Uses degrees (Lat/Lon). Great for recording locations, but terrible for measuring distance.
- **EPSG:3857 (Web Mercator)**: Used by Google Maps. Preserves shapes perfectly (so roads look correct), but massively distorts size (making Greenland look as big as Africa). Uses meters.

*Rule of Thumb*: Always ensure all your datasets are converted to the *same* CRS before trying to merge or plot them!

---

## 3. GeoPandas: Pandas for Maps

`GeoPandas` extends the famous `pandas` library to handle spatial data.

Instead of a standard `DataFrame`, it uses a `GeoDataFrame`. The difference is a special `geometry` column that holds Shapely objects (Points, Polygons).

With this single column, `GeoPandas` can perform complex geometric math instantly:
- `df.area`: Calculates the area of every polygon in the dataframe.
- `df.distance(point)`: Calculates the distance from every shape to a specific point.
- `df.plot()`: Automatically draws a beautiful map of the shapes!

---

## 4. Spatial Joins & Buffer Zones

In standard SQL/Pandas, you join two tables if they share an exact ID (e.g., `user_id`).
In Geospatial Analysis, you do **Spatial Joins**. You join two tables based on their physical intersection!

Example: You have a table of 10,000 Uber Dropoff Points, and a table of 50 City Neighborhood Polygons.
A Spatial Join (`gpd.sjoin`) will instantly add the correct "Neighborhood Name" to every single Uber dropoff, simply by calculating which Polygon each Point falls inside.

**Buffer Zones**:
If you want to find all coffee shops within 1 mile of a subway station, you use `.buffer(1_mile)` on the station Point. This turns the Point into a circular Polygon. Then, you spatially join the coffee shops that intersect with that circle!

---

## 5. Interactive Mapping with Folium

Static maps (via `matplotlib`) are great for reports, but stakeholders love interactive maps they can zoom and click.

`Folium` is a Python library that generates interactive **Leaflet.js** maps directly inside Jupyter Notebooks.

You can easily create:
- **Choropleth Maps**: Polygons colored based on a variable (e.g., coloring states by average income).
- **Heatmaps**: Visualizing density of points (e.g., where crimes occur most frequently in a city).
- **Markers**: Interactive pins with popup text.

---

## 6. Library Implementation (GeoPandas & Folium)

Let's do a quick spatial analysis: Finding the distance between two cities and plotting them.

```python
import geopandas as gpd
from shapely.geometry import Point
import folium

# 1. Create a GeoDataFrame of Cities
cities_data = {
    'City': ['New York', 'Los Angeles', 'Chicago'],
    'Latitude': [40.7128, 34.0522, 41.8781],
    'Longitude': [-74.0060, -118.2437, -87.6298]
}

# Create Shapely Point objects (Note: Shapely expects Longitude first, then Latitude)
geometry = [Point(lon, lat) for lon, lat in zip(cities_data['Longitude'], cities_data['Latitude'])]
gdf = gpd.GeoDataFrame(cities_data, geometry=geometry, crs="EPSG:4326")

print("Initial GeoDataFrame (Lat/Lon):")
print(gdf[['City', 'geometry']])

# 2. Distance Calculation
# To measure distance in meters, we MUST convert from GPS (EPSG:4326) 
# to a projected CRS. We'll use EPSG:3857 (Web Mercator)
gdf_projected = gdf.to_crs("EPSG:3857")

# Calculate distance between NY (index 0) and LA (index 1)
ny_geom = gdf_projected.loc[0, 'geometry']
la_geom = gdf_projected.loc[1, 'geometry']

distance_meters = ny_geom.distance(la_geom)
print(f"\nDistance NY to LA: {distance_meters / 1000:.2f} kilometers")
# Note: Web Mercator distorts distances; for perfect accuracy, use a local CRS 
# or the Haversine formula on the unprojected coordinates!

# 3. Create an Interactive Folium Map
# Center map on the US
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add markers for each city
for idx, row in gdf.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['City'],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# Save the interactive map to an HTML file
m.save("us_cities_map.html")
print("\nMap saved to 'us_cities_map.html'. Open it in a browser!")
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Choropleth of Unemployment**: Download a shapefile of US Counties and a CSV of county unemployment rates. Use `GeoPandas` to merge them (on county FIPS code) and use `.plot(column='rate', cmap='OrRd')` to create a beautiful heat-map of unemployment across the USA.
- 🟡 **Airbnb Density Heatmap**: Download the free Airbnb listings dataset for a major city (e.g., London). Use `folium.plugins.HeatMap` to generate an interactive heatmap of where the most expensive listings are concentrated. Add subway stations as interactive markers to see if expensive listings cluster near transit!

### What's Next
| Next | Why |
|------|-----|
| [Big Data & Distributed ML](./05-Big-Data-And-Distributed-ML.md) | We've mastered complex data structures. But what if your dataframe is 5 Terabytes and literally cannot fit into RAM? We must move from single-machine Python to Distributed Computing clusters. |

---

[← Survival Analysis](./03-Survival-Analysis.md) | [Back to Index](../README.md) | [Next: Big Data And Distributed ML →](./05-Big-Data-And-Distributed-ML.md)
