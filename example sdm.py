Species Distribution Modeling Visualization
Species: Mountain Lion (Puma concolor)
Region: USA

Creates 4-panel SDM output:
1. Observed Occurrences
2. Habitat Suitability
3. Predicted Distribution
4. Model Accuracy


import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier

 Load USA Map

world = gpd.read_file(
    gpd.datasets.get_path("naturalearth_lowres")
)

usa = world[world.name == "United States of America"]
 Generate Synthetic Occurrence Data


np.random.seed(42)

n_points = 400

# USA bounding box
lon = np.random.uniform(-125, -67, n_points)
lat = np.random.uniform(25, 49, n_points)

# Simulated presence data
presence = np.random.choice([0, 1], n_points)

X = np.column_stack((lon, lat))
y = presence

 Train SDM Model


model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X, y)
 Create Prediction Grid


grid_lon = np.linspace(-125, -67, 300)
grid_lat = np.linspace(25, 49, 200)

xx, yy = np.meshgrid(grid_lon, grid_lat)

grid_points = np.column_stack(
    (xx.ravel(), yy.ravel())
)

probabilities = model.predict_proba(
    grid_points
)[:, 1]

prob_map = probabilities.reshape(xx.shape)

# Binary distribution
predicted_distribution = (
    prob_map > 0.5
).astype(int)

# Fake accuracy map
accuracy_map = np.random.choice(
    [0, 1, 2],
    size=prob_map.shape
)



fig, axs = plt.subplots(
    2,
    2,
    figsize=(16, 12)
)

 Panel 1 — Observed Occurrences
usa.plot(
    ax=axs[0, 0],
    color="white",
    edgecolor="black"
)

axs[0, 0].scatter(
    lon[y == 1],
    lat[y == 1],
    c="black",
    s=8,
    label="Presence"
)

axs[0, 0].set_title(
    "Observed Occurrences"
)

 Panel 2 — Habitat Suitability
usa.plot(
    ax=axs[0, 1],
    color="white",
    edgecolor="black"
)

im1 = axs[0, 1].imshow(
    prob_map,
    extent=(-125, -67, 25, 49),
    origin="lower",
    cmap="jet",
    alpha=0.6
)

axs[0, 1].set_title(
    "Habitat Suitability"
)

plt.colorbar(
    im1,
    ax=axs[0, 1],
    label="Suitability"
)

 Panel 3 — Predicted Distribution
usa.plot(
    ax=axs[1, 0],
    color="white",
    edgecolor="black"
)

im2 = axs[1, 0].imshow(
    predicted_distribution,
    extent=(-125, -67, 25, 49),
    origin="lower",
    cmap="Greens",
    alpha=0.6
)

axs[1, 0].set_title(
    "Predicted Distribution"
)

 Panel 4 — Model Accuracy
usa.plot(
    ax=axs[1, 1],
    color="white",
    edgecolor="black"
)

im3 = axs[1, 1].imshow(
    accuracy_map,
    extent=(-125, -67, 25, 49),
    origin="lower",
    cmap="RdYlGn",
    alpha=0.6
)

axs[1, 1].set_title(
    "Model Accuracy"
)

 Main Title
plt.suptitle(
    "Species Distribution Modeling: Mountain Lion (Puma concolor) — USA",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

 Save Figure
plt.savefig(
    "puma_concolor_sdm_usa.png",
    dpi=300
)

plt.show()
