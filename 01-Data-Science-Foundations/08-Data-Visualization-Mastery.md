# 🎨 Data Visualization Mastery — Telling Stories with Data

> **Prerequisites**: Exploratory Data Analysis | **Difficulty**: ⭐⭐☆☆☆ Beginner-Intermediate

---

## 📋 Table of Contents

1. [The Philosophy of Data Visualization](#1-the-philosophy-of-data-visualization)
2. [Matplotlib: The Foundation](#2-matplotlib-the-foundation)
3. [Seaborn: Statistical Visualization](#3-seaborn-statistical-visualization)
4. [Plotly: Interactive Dashboards](#4-plotly-interactive-dashboards)
5. [Choosing the Right Chart](#5-choosing-the-right-chart)
6. [The Grammar of Graphics](#6-the-grammar-of-graphics)
7. [Color Theory and Accessibility](#7-color-theory-and-accessibility)
8. [Advanced Techniques: Geospatial & 3D](#8-advanced-techniques-geospatial--3d)
9. [Project Ideas & What's Next](#9-project-ideas--whats-next)

---

## 1. The Philosophy of Data Visualization

Data visualization is not just about making pretty pictures; it is about **communication**. A good visualization reduces cognitive load, allowing the viewer to instantly grasp patterns, trends, and anomalies that would take hours to extract from a raw table of numbers.

### The Goals of Visualization

1. **Exploration**: Finding unknown patterns during the EDA phase. (Usually quick, dirty, and done for yourself).
2. **Confirmation**: Verifying a hypothesis or statistical model. (e.g., checking residuals in linear regression).
3. **Presentation**: Communicating insights to stakeholders. (Must be polished, clear, and annotated).

### Principles of Good Design (Edward Tufte)

- **Maximize the Data-to-Ink Ratio**: Remove non-data ink (heavy gridlines, borders, backgrounds) that doesn't convey information.
- **Avoid "Chartjunk"**: Eliminate 3D effects, shadows, and unnecessary decorations.
- **Tell the Truth**: Don't distort axes (e.g., bar charts MUST start at 0) or use misleading perspectives.

---

## 2. Matplotlib: The Foundation

Matplotlib is the granddaddy of Python visualization. While it can be verbose, it gives you **pixel-perfect control** over every element of a figure.

### The Object-Oriented Interface

There are two ways to use Matplotlib: the MATLAB-style state-machine (`plt.plot()`) and the Object-Oriented (OO) interface (`fig, ax = plt.subplots()`). **Always use the OO interface** for complex plots.

- **Figure**: The overall window or page that everything is drawn on.
- **Axes**: The actual plot (the data area, ticks, labels). A Figure can have many Axes.

### Comprehensive Matplotlib Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
noise = np.random.normal(0, 0.2, 100)

# Create a Figure and multiple Axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

# --- Subplot 1: Line plot with error bands ---
ax1.plot(x, y1, color='#2ca02c', linewidth=2.5, label='Signal 1 (Sin)')
ax1.plot(x, y2, color='#1f77b4', linewidth=2.5, linestyle='--', label='Signal 2 (Cos)')

# Add confidence intervals
ax1.fill_between(x, y1 - 0.3, y1 + 0.3, color='#2ca02c', alpha=0.2)
ax1.fill_between(x, y2 - 0.3, y2 + 0.3, color='#1f77b4', alpha=0.2)

# Annotations
peak_x = x[np.argmax(y1)]
peak_y = np.max(y1)
ax1.annotate('Maximum Output', xy=(peak_x, peak_y), xytext=(peak_x + 1, peak_y + 0.2),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=12, fontweight='bold')

# Styling
ax1.set_title('Signal Processing Analysis', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Time (ms)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right', frameon=True, shadow=True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Subplot 2: Histogram of noise ---
ax2.hist(noise, bins=15, color='#d62728', edgecolor='white', alpha=0.8)
ax2.axvline(np.mean(noise), color='black', linestyle='--', linewidth=2, label=f'Mean: {np.mean(noise):.2f}')

ax2.set_title('Noise Distribution', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Error Value', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Adjust layout and save
plt.tight_layout()
plt.savefig('matplotlib_masterclass.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 3. Seaborn: Statistical Visualization

Seaborn is built on top of Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. It natively understands Pandas DataFrames and automatically calculates statistics (like confidence intervals).

### Why use Seaborn?
1. Built-in themes and color palettes.
2. Direct DataFrame integration.
3. Automatically aggregates and plots statistical relationships.

### Seaborn Gallery

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load built-in dataset
tips = sns.load_dataset("tips")

# Set global theme
sns.set_theme(style="whitegrid", palette="pastel")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Violin Plot (Distribution + Boxplot)
# Great for seeing the full shape of the data across categories
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker", split=True, 
               inner="quart", linewidth=1.5, ax=axes[0, 0])
axes[0, 0].set_title('Total Bill Distribution by Day & Smoker Status', fontweight='bold')

# 2. Scatterplot with Regression Line (lmplot/regplot)
# Automatically calculates and plots a linear regression with 95% CI
sns.regplot(data=tips, x="total_bill", y="tip", scatter_kws={'alpha':0.6}, 
            line_kws={'color': 'red'}, ax=axes[0, 1])
axes[0, 1].set_title('Tip vs. Total Bill (with Trendline)', fontweight='bold')

# 3. Correlation Heatmap
# Essential for feature engineering
corr = tips[['total_bill', 'tip', 'size']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True, 
            linewidths=2, cbar_kws={"shrink": .8}, ax=axes[1, 0])
axes[1, 0].set_title('Feature Correlation Matrix', fontweight='bold')

# 4. Joint Plot (Simulated using scatter + kde)
# Shows joint distribution and marginal distributions
sns.kdeplot(data=tips, x="total_bill", y="tip", fill=True, cmap="mako", 
            alpha=0.8, ax=axes[1, 1])
sns.scatterplot(data=tips, x="total_bill", y="tip", color="white", 
                alpha=0.5, s=20, ax=axes[1, 1])
axes[1, 1].set_title('2D Density of Tip vs Bill', fontweight='bold')

plt.tight_layout()
plt.savefig('seaborn_gallery.png', dpi=300)
plt.show()
```

### Pairplot: The Ultimate EDA Tool

When exploring a new dataset, `sns.pairplot()` is often the very first function you should call. It creates a grid of scatterplots for every numerical variable pair, and histograms for the diagonals.

```python
# This single line generates a massive grid comparing all numerical features
# colored by a categorical variable.
sns.pairplot(tips, hue="sex", corner=True, height=2.5)
```

---

## 4. Plotly: Interactive Dashboards

While Matplotlib and Seaborn generate static images, Plotly generates **interactive HTML/JS visualizations**. Users can hover over data points, zoom, pan, and click legends to toggle series.

### Plotly Express

Plotly Express (`px`) is the easy-to-use, high-level interface to Plotly (similar to what Seaborn is to Matplotlib).

```python
import plotly.express as px
import pandas as pd

# Load dataset
df = px.data.gapminder()

# Filter for the year 2007
df_2007 = df[df.year == 2007]

# Create a highly interactive bubble chart
fig = px.scatter(
    df_2007, 
    x="gdpPercap", 
    y="lifeExp", 
    size="pop",               # Bubble size represents population
    color="continent",        # Color by continent
    hover_name="country",     # Tooltip shows country name
    log_x=True,               # GDP is highly skewed, use log scale
    size_max=60, 
    title="Global Wealth and Health (2007)",
    template="plotly_dark"    # Built-in dark mode
)

# Update layout for presentation
fig.update_layout(
    xaxis_title="GDP per Capita (Log Scale, USD)",
    yaxis_title="Life Expectancy (Years)",
    font=dict(family="Inter, sans-serif", size=14)
)

# Saves as an interactive HTML file that can be opened in any browser
fig.write_html("plotly_gapminder.html")
# fig.show()  # Display in Jupyter Notebook
```

### Plotly Graph Objects

For extreme customization or complex subplots, use `plotly.graph_objects` (`go`).

```python
import plotly.graph_objects as go

fig = go.Figure()

# Add a candlestick chart for financial data
fig.add_trace(go.Candlestick(
    x=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    open=[100, 105, 102, 108, 110],
    high=[110, 108, 109, 115, 112],
    low=[95, 100, 98, 105, 100],
    close=[105, 102, 108, 110, 105],
    name="Stock XYZ"
))

fig.update_layout(
    title='Stock Price Movement',
    yaxis_title='Price (USD)',
    xaxis_rangeslider_visible=False
)
```

---

## 5. Choosing the Right Chart

Choosing the wrong chart can obscure insights or actively mislead the viewer.

### 1. Showing Change Over Time
- **Line Chart**: The standard for time series.
- **Area Chart**: Good for showing cumulative totals over time.
- ❌ *Don't use Bar charts for continuous time.*

### 2. Showing Comparison
- **Bar Chart**: Comparing categorical data. (Always start y-axis at 0!).
- **Grouped/Stacked Bar**: Comparing sub-categories.
- **Radar/Spider Chart**: Comparing multiple variables for a single entity (e.g., player stats).

### 3. Showing Distribution
- **Histogram**: Continuous data distribution.
- **Box Plot**: Shows median, quartiles, and outliers. Great for comparing distributions across categories.
- **Violin Plot**: Combines box plot with KDE (density). Shows multimodal distributions perfectly.

### 4. Showing Correlation / Relationship
- **Scatter Plot**: Relationship between two continuous variables.
- **Bubble Chart**: Scatter plot where size encodes a third variable.
- **Heatmap**: Showing correlation matrices or 2D density.

### 5. Showing Composition
- **Stacked Bar Chart**: Best for showing part-to-whole relationships over time or across categories.
- **Treemap**: Hierarchical part-to-whole (better than Pie charts for many categories).
- ❌ *Avoid Pie Charts*: Humans are terrible at judging angles. Only use if comparing exactly 2-3 slices where the difference is massive.

---

## 6. The Grammar of Graphics

The "Grammar of Graphics" (Leland Wilkinson) is a framework that treats visualizations as being built from independent, underlying components. It is the philosophy behind R's `ggplot2` and Python's `plotnine`.

Every chart is composed of:
1. **Data**: The raw dataset.
2. **Aesthetics (aes)**: Mapping variables to visual properties (x-axis, y-axis, color, size, shape).
3. **Geometries (geom)**: The actual shapes drawn (points, lines, bars).
4. **Facets**: Creating subplots based on categorical variables.
5. **Statistics**: Transformations like binning or regression.
6. **Coordinates**: Cartesian, polar, map projections.
7. **Themes**: Fonts, background colors, gridlines.

```python
# Using Plotnine (Python's ggplot2)
from plotnine import *
import pandas as pd
import seaborn as sns

mtcars = sns.load_dataset("mpg")
mtcars = mtcars.dropna()

# Building a plot layer by layer using the Grammar of Graphics
plot = (
    ggplot(mtcars, aes(x='weight', y='mpg', color='origin', size='cylinders'))
    + geom_point(alpha=0.7)                    # The geometric object
    + stat_smooth(method='lm', se=False)       # Statistical transformation
    + facet_wrap('~origin')                    # Subplots
    + theme_minimal()                          # Clean theme
    + labs(title="Fuel Efficiency vs Weight", 
           x="Weight (lbs)", y="Miles per Gallon")
)

# plot.draw()
```

---

## 7. Color Theory and Accessibility

Color is the most misused visual channel. 

### Types of Color Palettes
1. **Categorical (Qualitative)**: Distinct colors for distinct groups (e.g., Blue for US, Red for UK, Green for Japan).
2. **Sequential**: Gradients for numeric data moving in one direction (e.g., Light blue to Dark blue for population density).
3. **Diverging**: Gradients representing numeric data with a meaningful midpoint (e.g., Red for negative profits, White for 0, Blue for positive profits).

### Best Practices
- **Colorblindness**: 8% of men are red-green colorblind. Never rely solely on red vs. green to convey meaning. Use viridis, cividis, or magma (perceptually uniform and colorblind-safe).
- **Redundant Encoding**: If you use color to distinguish groups, also use different marker shapes (e.g., circles vs squares).
- **Contrast**: Ensure text has high contrast against backgrounds.

```python
# Seaborn has excellent built-in palettes
import seaborn as sns

# View colorblind safe categorical palette
sns.color_palette("colorblind")

# View perceptually uniform sequential palette
sns.color_palette("viridis", as_cmap=True)

# View diverging palette centered at zero
sns.color_palette("vlag", as_cmap=True)
```

---

## 8. Advanced Techniques: Geospatial & 3D

### Geospatial Data with Folium

For mapping data to physical geography, `folium` (built on Leaflet.js) is exceptional.

```python
import folium

# Create a base map centered on New York
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='CartoDB dark_matter')

# Add a marker
folium.CircleMarker(
    location=[40.7580, -73.9855], # Times Square
    radius=10,
    popup='Times Square Traffic: High',
    color='#3186cc',
    fill=True,
    fill_color='#3186cc'
).add_to(m)

# Save as interactive map
# m.save("nyc_map.html")
```

### 3D Visualization

While often discouraged because perspective can distort data, 3D is useful for surfaces (e.g., loss landscapes in deep learning) or 3-variable spatial data.

```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate a 3D surface (e.g., a loss function landscape)
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_title('Loss Function Landscape Optimization', fontweight='bold')
ax.set_xlabel('Weight 1')
ax.set_ylabel('Weight 2')
ax.set_zlabel('Loss')

fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
```

---

## 9. Project Ideas & What's Next

### Project Ideas

#### 🟢 Project 1: The EDA Report Card (Beginner)
Take a dataset from Kaggle (like the Titanic or Housing Prices). Create a Jupyter notebook that uses Matplotlib and Seaborn to tell a complete story from raw data to insights. Ensure all charts have proper titles, labels, and colorblind-safe palettes.

#### 🟡 Project 2: Interactive Financial Dashboard (Intermediate)
Use Plotly to build an interactive dashboard of stock market data. Include candlestick charts, volume bar charts, and moving average overlays. Export the final result as an interactive HTML file.

#### 🔴 Project 3: COVID-19 Global Tracker (Advanced)
Combine pandas, Plotly, and Folium to create a geospatial dashboard tracking a metric (like cases or vaccination rates) over time across the globe. Implement a time-slider in Plotly so the user can watch the map change chronologically.

### What's Next

| Next Topic | Why |
|------------|-----|
| [Statistical Inference](./03-Statistical-Inference.md) | Use visualizations to understand statistical significance. |
| [Machine Learning Foundations](./01-What-Is-Data-Science-And-ML.md) | Move from exploring data to predicting outcomes. |
| [Feature Engineering](./07-Feature-Engineering.md) | Visualizations will tell you what new features need to be created. |

---

[← Feature Engineering](./07-Feature-Engineering.md) | [Back to Index](../README.md) | [Next: Feature Selection Methods →](./09-Feature-Selection-Methods.md)
