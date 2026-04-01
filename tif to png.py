import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def convert_tif_to_png(tif_path, png_path, cmap="viridis"):
    # Open the raster
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        # Mask no-data values
        data = np.ma.masked_equal(data, src.nodata)
    
    # Plot with color map
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap=cmap)
    plt.colorbar(label="Habitat Suitability (0-1)")
    plt.axis("off")
    plt.title("Species Habitat Suitability Map")
    
    # Save as PNG
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"PNG map saved at {png_path}")

def main():
    base_dir = ".."  # adjust as needed
    tif_path = os.path.join(base_dir, "outputs", "suitability_map.tif")
    png_path = os.path.join(base_dir, "outputs", "images", "suitability_map.png")
    
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    convert_tif_to_png(tif_path, png_path)

if __name__ == "__main__":
    main()
