# Fungi Image Clustering

This script clusters Instagram fungi posts based on visual properties using Google Cloud Vision API.

## Prerequisites

1. Python 3.7+
2. Google Cloud Vision API credentials
3. Input data:
   - `fungi_metadata.csv`: CSV file containing Instagram post metadata
   - `images/` directory containing all referenced images

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud Vision API:
   - Create a Google Cloud project
   - Enable the Vision API
   - Create and download service account credentials
   - Set the environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

## Directory Structure
```
.
├── fungi_metadata.csv
├── images/
│   └── [your image files]
├── cluster_fungi.py
└── output/
    ├── extracted_features.csv
    ├── reduced_features.csv
    ├── clustered_data.csv
    └── clusters.html
```

## Usage

1. Place your input CSV file as `fungi_metadata.csv` in the project root
2. Place all images in the `images/` directory
3. (Optional) Adjust the `NUM_CLUSTERS` variable in `cluster_fungi.py` if you want a different number of clusters
4. Run the script:
```bash
python cluster_fungi.py
```

## Categories

The script classifies posts into four main categories:

1. **Natural**: Photos of fungi in their natural environment
   - Wild mushrooms in forests, woods, trails
   - Natural landscapes and ecosystems
   - In-situ documentation

2. **Stylized**: Artistic photography of fungi
   - Macro photography
   - Bokeh effects
   - Artistic compositions
   - Professional lighting

3. **Staged**: Controlled or artificial settings
   - Studio shots
   - Product photography
   - Indoor arrangements
   - Tattoos and body art
   - Commercial content

4. **Symbolic**: Artistic interpretations
   - Illustrations
   - Digital art
   - Paintings
   - AI-generated content
   - Crafts and handmade items

## Output

The script generates several files in the `output/` directory:

1. Raw Data:
   - `extracted_features.csv`: Raw features from Vision API
   - `image_labels.json`: Detected labels for each image
   - `image_objects.json`: Detected objects in each image
   - `image_web_entities.json`: Web entities associated with images

2. Group Visualizations:
   - `groups.html`: All images organized by category
   - `groups_quick.html`: Quick view of each category

3. Category-Specific Results:
   Each category (natural, stylized, staged, symbolic) has its own subdirectory containing:
   - `pca_variance.csv`: PCA analysis results
   - `cluster_analysis.csv`: Detailed cluster statistics
   - `clusters.html`: Full cluster visualization
   - `clusters_quick.html`: Representative images from each cluster

### Quick View Selection Process

The quick view (`clusters_quick.html`) intelligently selects representative images from each cluster using the following process:

1. **Small Clusters**: If a cluster has 20 or fewer images, all images are shown.

2. **Large Clusters**: For clusters with more than 20 images, the selection is based on distance from the cluster center:
   - Calculate the cluster center by taking the mean of all feature vectors
   - Measure how far each image is from the center
   - Sort images by their distance

3. **Stratified Sampling**: To ensure diverse representation, the selection:
   - Divides the cluster into 4 distance ranges
   - Takes 5 samples from each range (20 total = 4 ranges × 5 samples)
   - This ensures we get:
     - Very typical examples (closest to center)
     - Somewhat typical examples (moderately close)
     - Somewhat unusual examples (moderately far)
     - Edge cases (furthest from center)

4. **Gap Filling**: If some ranges have fewer than 5 images, additional images are randomly selected from the remaining pool to reach 20 samples.

This approach ensures that:
- Users see representative examples from all parts of the cluster
- Both typical and unusual cases are represented
- The selection is balanced across different "regions" of the cluster

For example, in a mushroom cluster, this might show:
- Very typical mushroom photos (center)
- Slightly different angles/lighting (middle ranges)
- More unusual or edge cases (outer range)
