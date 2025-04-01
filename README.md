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

## Classification Process

The script uses a sophisticated multi-factor classification system to categorize posts into four main categories. Each post is evaluated based on:

1. **Vision API Labels**: Primary visual elements detected in the image
2. **Detected Objects**: Specific objects identified in the image
3. **Caption Text**: Post text and hashtags
4. **Web Entities**: Additional context from similar web images

### Classification Categories

1. **Natural**: Photos of fungi in their natural environment
   - **Labels**: wild, forest, nature, outdoor, natural, woods, trail
   - **Objects**: Tree, Plant, Grass, Mushroom, Fungus, Ground, Soil
   - **Context**: Found in natural settings, growing wild, in-situ documentation
   - **Boost**: Scientific terms (species, taxonomy, specimen)

2. **Stylized**: Artistic photography of fungi
   - **Labels**: macro, bokeh, depth of field, photography, artistic
   - **Objects**: Mushroom, Fungus, Plant (with artistic composition)
   - **Context**: Focus on aesthetic qualities and photographic techniques
   - **Boost**: Specific photo techniques (macro photography, bokeh)

3. **Staged**: Controlled or artificial settings
   - **Labels**: studio, product, indoor, commercial, lifestyle
   - **Objects**: Table, Cup, Bowl, Furniture, Crystal, Bottle
   - **Context**: Commercial products, indoor setups, controlled environments
   - **Emphasis**: Product presentation and commercial intent

4. **Symbolic**: Artistic interpretations
   - **Labels**: art, illustration, digital, painting, jewelry, tattoo
   - **Objects**: Art, Artwork, Painting, Jewelry, Body Art
   - **Context**: Creative interpretations and artistic expressions
   - **Boost**: AI-generated content, jewelry items, tattoo art

### Scoring System

Each post receives a score for each category based on:
1. Vision API labels (2.0 points per match)
2. Detected objects (1.5 points per match)
3. Caption text and hashtags (1.0 point per match)
4. Category-specific boosts (5.0 points for special features)

The post is assigned to the category with the highest final score.

## Clustering Process

After categorization, posts within each category are clustered using the following process:

1. **Feature Extraction**:
   - Uses Google Cloud Vision API to extract rich visual features
   - Captures color, texture, shape, and compositional elements
   - Generates high-dimensional feature vectors

2. **Dimensionality Reduction**:
   - Applies PCA (Principal Component Analysis)
   - Preserves 95% of variance while reducing dimensions
   - Makes clustering more efficient and robust

3. **K-Means Clustering**:
   - Applies k-means algorithm to the reduced features
   - Groups similar images based on their visual properties
   - Default: 10 clusters per category

4. **Cluster Analysis**:
   - Analyzes each cluster's characteristics:
     - Common Vision API labels
     - Frequent hashtags
     - Visual themes and patterns
   - Generates cluster summaries with representative images

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
