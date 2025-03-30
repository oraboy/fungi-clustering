import os
import json
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from google.cloud import vision
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from jinja2 import Template
import base64
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings('ignore')

# Configuration
DEFAULT_NUM_CLUSTERS = 10  # Default number of clusters if not specified
CSV_INPUT_PATH = 'fungi_metadata.csv'
IMAGES_DIR = 'images'
DEFAULT_OUTPUT_DIR = 'output'  # Default output directory if not specified
CACHE_DIR = 'output'  # Directory for caching Vision API results

# Filtering
FILTER_HASHTAGS = {
    'fungirl' , 'fungirls' , 'fungifts', 'fungift',
    'fungiethedolphin'      # commercial content
}

FILTER_LABELS = {
    'Egg',  'Footwear'             # inappropriate content
}

# Post classification categories and their associated terms
POST_CATEGORIES = {
    'natural': {
        'labels': {'wild', 'forest', 'nature', 'outdoor', 'natural', 'woods', 'trail', 'garden',
                  'scientific', 'species', 'taxonomy', 'identification', 'mycology', 'biology'},
        'objects': {'Tree', 'Plant', 'Grass', 'Mushroom', 'Fungus', 'Ground', 'Soil'},
        'text_markers': {'species', 'found', 'identified', 'specimen', 'habitat', 'wild'},
    },
    'stylized': {
        'labels': {'macro', 'bokeh', 'depth of field', 'photography', 'artistic', 'composition',
                  'light', 'shadow', 'mood', 'atmosphere', 'ethereal', 'mystical'},
        'objects': {'Mushroom', 'Fungus', 'Plant', 'Tree'},
        'text_markers': {'magical', 'beautiful', 'stunning', 'enchanted', 'fairy', 'mystical'},
    },
    'staged': {
        'labels': {'studio', 'product', 'indoor', 'artificial light', 'commercial', 'lifestyle',
                  'food', 'drink', 'setup', 'arrangement', 'display'},
        'objects': {'Table', 'Cup', 'Bowl', 'Person', 'Human', 'Furniture', 'Crystal', 'Bottle'},
        'text_markers': {'product', 'shop', 'buy', 'available', 'store', 'healing', 'wellness'},
    },
    'symbolic': {
        'labels': {'art', 'illustration', 'digital', 'painting', 'drawing', 'cartoon', 'graphic',
                  'fantasy', 'psychedelic', 'surreal', 'abstract', 'generated'},
        'objects': {'Art', 'Artwork', 'Painting', 'Drawing'},
        'text_markers': {'ai', 'generated', 'artwork', 'design', 'creative', 'trippy', 'psychedelic'},
    },
}

# Categories for cluster theme detection
CLUSTER_CATEGORIES = {
    'artwork': {'art', 'drawing', 'painting', 'illustration', 'sketch', 'digital', 'artistic', 'artwork'},
    'nature': {'wild', 'forest', 'nature', 'outdoor', 'natural', 'woods', 'trail', 'garden'},
    'microscopic': {'microscope', 'microscopic', 'spores', 'cells', 'scientific', 'lab', 'research'},
    'food': {'cooking', 'food', 'recipe', 'culinary', 'kitchen', 'dish', 'meal', 'edible'},
    'educational': {'education', 'learning', 'study', 'research', 'science', 'biology', 'mycology'},
    'photography': {'photo', 'photography', 'camera', 'macro', 'closeup', 'portrait'}
}

def setup_directories(output_dir):
    """Create output directory and category subdirectories if they don't exist"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each category
    for category in ['natural', 'stylized', 'staged', 'symbolic']:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)

def get_vision_client():
    """Create an authenticated Vision API client"""
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(
        'credentials/has-tags-1-cec63ad10c6d.json'
    )
    return vision.ImageAnnotatorClient(credentials=credentials)

def extract_image_features(image_path):
    """Extract features from image using Google Cloud Vision API"""
    client = get_vision_client()
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Get various features from the API
    response = client.annotate_image({
        'image': image,
        'features': [
            {'type_': vision.Feature.Type.LABEL_DETECTION},
            {'type_': vision.Feature.Type.IMAGE_PROPERTIES},
            {'type_': vision.Feature.Type.OBJECT_LOCALIZATION},
            {'type_': vision.Feature.Type.WEB_DETECTION},
        ]
    })
    
    # Extract color properties
    colors = response.image_properties_annotation.dominant_colors.colors
    color_features = []
    for color in colors[:3]:  # Take top 3 dominant colors
        color_features.extend([
            color.color.red / 255.0,
            color.color.green / 255.0,
            color.color.blue / 255.0,
            color.score,
            color.pixel_fraction  # Add pixel coverage
        ])
    
    # Pad color features if less than 3 colors
    while len(color_features) < 15:  # Now 5 features per color
        color_features.extend([0, 0, 0, 0, 0])
    
    # Extract label scores
    label_scores = [label.score for label in response.label_annotations[:5]]
    while len(label_scores) < 5:
        label_scores.append(0)
    
    # Extract object localization features
    objects = response.localized_object_annotations
    object_features = []
    for obj in objects[:3]:  # Take top 3 objects
        # Calculate object area from bounding box
        vertices = obj.bounding_poly.normalized_vertices
        width = vertices[1].x - vertices[0].x
        height = vertices[2].y - vertices[1].y
        area = width * height
        object_features.extend([obj.score, area])
    
    # Pad object features
    while len(object_features) < 6:  # 2 features per object
        object_features.extend([0, 0])
    
    # Extract web detection features
    web = response.web_detection
    web_features = [
        len(web.web_entities) / 10.0,  # Normalize number of web entities
        len(web.full_matching_images) / 10.0,  # Normalize number of matching images
        len(web.visually_similar_images) / 10.0,  # Normalize number of similar images
        1.0 if any('AI' in entity.description.upper() or 'GENERATED' in entity.description.upper()
                  for entity in web.web_entities) else 0.0  # AI generation detection
    ]
    
    # Combine all features
    features = color_features + label_scores + object_features + web_features
    
    # Get labels for filtering and display
    labels = [label.description for label in response.label_annotations]
    
    # Get objects for classification
    objects = [{
        'name': obj.name,
        'score': obj.score,
        'area': width * height
    } for obj in response.localized_object_annotations]
    
    # Get web entities for classification
    web_entities = [{
        'description': entity.description,
        'score': entity.score
    } for entity in web.web_entities]
    
    # Add web entities to labels for better classification
    web_labels = [entity['description'] for entity in web_entities if entity['score'] > 0.5]
    labels.extend(web_labels)
    
    return features, labels, objects, web_entities

def clean_filename(url):
    """Extract clean filename from Instagram URL"""
    # Handle pandas Series
    if isinstance(url, pd.Series):
        url = url.iloc[0] if len(url) > 0 else ''
    # Get the base filename
    filename = os.path.basename(url)
    # Remove query parameters
    clean_name = filename.split('?')[0]
    return clean_name

def load_cached_features():
    """Load cached features, labels, objects, and web entities from previous run"""
    features_path = os.path.join(CACHE_DIR, 'extracted_features.csv')
    labels_path = os.path.join(CACHE_DIR, 'image_labels.json')
    objects_path = os.path.join(CACHE_DIR, 'image_objects.json')
    web_path = os.path.join(CACHE_DIR, 'image_web_entities.json')
    
    if all(os.path.exists(p) for p in [features_path, labels_path, objects_path, web_path]):
        features_df = pd.read_csv(features_path, index_col=0)
        with open(labels_path, 'r') as f:
            image_labels = json.load(f)
        with open(objects_path, 'r') as f:
            image_objects = json.load(f)
        with open(web_path, 'r') as f:
            image_web_entities = json.load(f)
        return features_df, image_labels, image_objects, image_web_entities
    return None, None, None, None

def save_features_and_labels(features_df, image_labels, image_objects, image_web_entities):
    """Save features, labels, objects, and web entities to cache"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    features_df.to_csv(os.path.join(CACHE_DIR, 'extracted_features.csv'))
    with open(os.path.join(CACHE_DIR, 'image_labels.json'), 'w') as f:
        json.dump(image_labels, f)
    with open(os.path.join(CACHE_DIR, 'image_objects.json'), 'w') as f:
        json.dump(image_objects, f)
    with open(os.path.join(CACHE_DIR, 'image_web_entities.json'), 'w') as f:
        json.dump(image_web_entities, f)

def process_images(df, refresh_vision=False):
    """Process all images and extract features and classify posts"""
    # Create a unique index based on image filenames
    df.index = [clean_filename(url) for url in df['image_url']]
    
    if not refresh_vision:
        # Try to load cached results
        features_df, image_labels, image_objects, image_web_entities = load_cached_features()
        if features_df is not None:
            print("Using cached features and labels from previous run")
            # Check if we have all the features we need
            missing_files = set(df.index) - set(features_df.index)
            if len(missing_files) > 0:
                print(f"\nMissing features for {len(missing_files)} files. Running Vision API for these files...")
                # Process only missing files
                missing_features = []
                missing_labels = {}
                missing_objects = {}
                missing_web_entities = {}
                for idx in missing_files:
                    image_filename = clean_filename(df.loc[idx, 'image_url'])
                    image_path = os.path.join(IMAGES_DIR, image_filename)
                    if not os.path.exists(image_path):
                        print(f"Warning: Image file not found: {image_path}")
                        continue
                    
                    features, labels, objects, web_entities = extract_image_features(image_path)
                    missing_features.append(features)
                    missing_labels[idx] = labels
                    missing_objects[idx] = objects
                    missing_web_entities[idx] = web_entities
                
                # Add new features and labels
                missing_features_df = pd.DataFrame(missing_features, index=list(missing_files))
                features_df = pd.concat([features_df, missing_features_df])
                image_labels.update(missing_labels)
                
                # Save updated cache
                save_features_and_labels(features_df, image_labels, missing_objects, missing_web_entities)
            
            # Now get only the features we need
            features_df = features_df.loc[df.index]
            return features_df, image_labels, image_objects, image_web_entities
    
    features_list = []
    processed_files = []
    image_labels = {}
    image_objects = {}
    image_web_entities = {}
    post_categories = {}
    
    print("Processing images with Google Vision API...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Extract clean filename from the URL
            image_filename = clean_filename(row['image_url'])
            image_path = os.path.join(IMAGES_DIR, image_filename)
            
            # Extract features, labels, objects, and web entities
            features, labels, objects, web_entities = extract_image_features(image_path)
            features_list.append(features)
            processed_files.append(image_filename)
            image_labels[image_filename] = labels
            image_objects[image_filename] = objects
            image_web_entities[image_filename] = web_entities
            
            # Classify the post
            category = classify_post(features, labels, objects, row.get('caption', ''), web_entities)
            post_categories[image_filename] = category
            
        except Exception as e:
            print(f"Error processing {image_path if 'image_path' in locals() else row['image_url']}: {str(e)}")
            continue
    
    features_df = pd.DataFrame(features_list, index=processed_files)
    
    # Add category to the DataFrame
    df['category'] = pd.Series(post_categories)
    
    # Save results to cache
    save_features_and_labels(features_df, image_labels, image_objects, image_web_entities)
    
    # Add category to the DataFrame
    df['category'] = pd.Series(post_categories)
    
    return features_df, image_labels, image_objects, image_web_entities

def filter_posts(df, image_labels):
    """Filter out posts based on hashtags and Vision API labels"""
    print("\nFiltering posts...")
    initial_count = len(df)
    
    # Track which hashtags and labels were found
    found_hashtags = {tag: 0 for tag in FILTER_HASHTAGS}
    found_labels = {label: 0 for label in FILTER_LABELS}
    
    # Filter by hashtags
    def get_filtered_hashtags(text):
        if not isinstance(text, str):
            return set()
        # Extract hashtags (words starting with #)
        post_hashtags = {tag.lower().strip('#') for tag in text.split() if tag.startswith('#')}
        return post_hashtags & FILTER_HASHTAGS
    
    # Find and count filtered hashtags
    filtered_by_hashtags = []
    for idx, row in df.iterrows():
        filtered_tags = get_filtered_hashtags(row['post_text'])
        if filtered_tags:
            filtered_by_hashtags.append(idx)
            for tag in filtered_tags:
                found_hashtags[tag] += 1
    
    df = df.drop(filtered_by_hashtags)
    
    # Filter by Vision API labels
    filtered_by_labels = []
    for img_name, labels in image_labels.items():
        filtered_labels = set(labels) & FILTER_LABELS
        if filtered_labels:
            filtered_by_labels.append(img_name)
            for label in filtered_labels:
                found_labels[label] += 1
    
    # Convert image URLs to filenames for filtering
    filtered_filenames = {clean_filename(url) for url in filtered_by_labels}
    filtered_by_labels = df[df['image_url'].apply(clean_filename).isin(filtered_filenames)].index
    df = df.drop(filtered_by_labels)
    
    # Print filtering stats
    print("\nFiltered hashtags:")
    for tag, count in found_hashtags.items():
        if count > 0:
            print(f"  #{tag}: {count} posts")
    
    print("\nFiltered Vision API labels:")
    for label, count in found_labels.items():
        if count > 0:
            print(f"  {label}: {count} posts")
    
    total_filtered = initial_count - len(df)
    print(f"\nTotal posts remaining: {len(df)} (removed {total_filtered} posts)")
    
    return df

def reduce_dimensions(features_df, output_dir):
    """Perform PCA on the features"""
    if len(features_df) == 0:
        print("No features to process. Please check if any images were successfully processed.")
        return None

    print("\nPerforming dimensionality reduction...")
    # Convert column names to strings
    features_df.columns = features_df.columns.astype(str)
    
    # Fill NaN values with 0
    features_df = features_df.fillna(0)
    
    pca = PCA(n_components=0.95)  # Preserve 95% of variance
    reduced_features = pca.fit_transform(features_df)
    
    reduced_df = pd.DataFrame(
        reduced_features,
        index=features_df.index,
        columns=[f'PC{i+1}' for i in range(reduced_features.shape[1])]
    )
    reduced_df.to_csv(os.path.join(output_dir, 'reduced_features.csv'))
    return reduced_df

def classify_post(features, labels, objects, caption, web_entities):
    """Classify a post into one of the four categories based on its features and content"""
    scores = {category: 0.0 for category in POST_CATEGORIES.keys()}
    
    # Convert everything to lowercase for matching
    labels = [l.lower() for l in labels]
    objects = [o['name'].lower() for o in objects]
    caption_words = caption.lower().split() if isinstance(caption, str) else []
    web_entities = [e['description'].lower() for e in web_entities]
    
    # Extract hashtags
    hashtags = {tag.lower().strip('#') for tag in caption_words 
               if tag.startswith('#') and tag.lower().strip('#') not in FILTER_HASHTAGS}
    
    for category, markers in POST_CATEGORIES.items():
        # Score based on Vision API labels
        label_matches = sum(1 for label in labels if label in markers['labels'])
        scores[category] += label_matches * 2.0  # Labels are strong indicators
        
        # Score based on detected objects
        object_matches = sum(1 for obj in objects if obj in markers['objects'])
        scores[category] += object_matches * 1.5
        
        # Score based on caption text and hashtags
        text_matches = sum(1 for word in caption_words if word in markers['text_markers'])
        text_matches += sum(1 for tag in hashtags if tag in markers['text_markers'])
        scores[category] += text_matches
        
        # Additional category-specific scoring
        if category == 'symbolic':
            # Check for AI-generated content
            if any('ai' in entity or 'generated' in entity for entity in web_entities):
                scores[category] += 5.0
        elif category == 'stylized':
            # Check for artistic photo techniques
            if any(term in labels for term in {'macro photography', 'bokeh', 'depth of field'}):
                scores[category] += 3.0
        elif category == 'natural':
            # Boost score if multiple scientific terms are present
            scientific_terms = {'species', 'taxonomy', 'specimen', 'habitat'}
            if sum(1 for term in scientific_terms if term in labels) > 1:
                scores[category] += 4.0
        elif category == 'staged':
            # Check for indoor/studio environment
            if any(term in labels for term in {'indoor', 'studio', 'artificial light'}):
                scores[category] += 3.0
    
    # Get the category with the highest score
    max_score = max(scores.values())
    if max_score == 0:
        return 'natural'  # Default to natural if no strong signals
    
    return max(scores.items(), key=lambda x: x[1])[0]

def extract_hashtags(text):
    """Extract hashtags from text, excluding filtered ones"""
    if not isinstance(text, str):
        return set()
    return {tag.lower().strip('#') for tag in text.split() 
            if tag.startswith('#') and tag.lower().strip('#') not in FILTER_HASHTAGS}

def get_cluster_themes(cluster_data, df, image_labels):
    print(f"\nAnalyzing cluster with {len(cluster_data)} images...")
    
    # The index is already the filename
    filenames = cluster_data.index.tolist()
    print(f"Sample filenames: {filenames[:3]}")
    """Analyze cluster content to determine main themes"""
    # Initialize counters for different sources of information
    label_counter = Counter()
    hashtag_counter = Counter()
    
    # Process each item in the cluster
    for idx in cluster_data.index:
        try:
            # Get row from original dataframe
            row = df.loc[idx]
            
            # Get the original filename and labels
            filename = clean_filename(row['image_url'])
            if filename in image_labels:
                labels = image_labels[filename]
                label_counter.update(labels)
                print(f"Labels for {filename}: {labels}")
            
            # Add hashtags from post text
            post_text = row['post_text']
            hashtags = extract_hashtags(post_text)
            hashtag_counter.update(hashtags)
        except KeyError:
            continue
    
    # Score each category based on labels and hashtags
    category_scores = defaultdict(float)
    
    # Print most common labels and hashtags
    print("\nTop 10 Vision API labels:")
    for label, count in label_counter.most_common(10):
        print(f"  {label}: {count} times")
    
    print("\nTop 10 hashtags:")
    for hashtag, count in hashtag_counter.most_common(10):
        print(f"  #{hashtag}: {count} times")
    
    # Process Vision API labels with partial matching
    total_labels = sum(label_counter.values()) or 1
    for label, count in label_counter.most_common():
        label_lower = label.lower()
        label_words = set(label_lower.split())
        
        for category, terms in CLUSTER_CATEGORIES.items():
            # Check for full term matches
            for term in terms:
                if term in label_lower:
                    category_scores[category] += (count / total_labels) * 1.0
                    break
            # Check for partial word matches
            else:
                term_words = set()
                for term in terms:
                    term_words.update(term.split())
                matching_words = label_words & term_words
                if matching_words:
                    category_scores[category] += (count / total_labels) * (len(matching_words) / len(label_words))
    
    # Process hashtags with partial matching
    total_hashtags = sum(hashtag_counter.values()) or 1
    for hashtag, count in hashtag_counter.most_common():
        hashtag_lower = hashtag.lower()
        hashtag_words = set(hashtag_lower.split())
        
        for category, terms in CLUSTER_CATEGORIES.items():
            # Check for full term matches
            for term in terms:
                if term in hashtag_lower:
                    category_scores[category] += (count / total_hashtags) * 2.0  # Weight hashtags more
                    break
            # Check for partial word matches
            else:
                term_words = set()
                for term in terms:
                    term_words.update(term.split())
                matching_words = hashtag_words & term_words
                if matching_words:
                    category_scores[category] += (count / total_hashtags) * (len(matching_words) / len(hashtag_words)) * 2.0
    
    # Print category scores
    print("\nCategory scores:")
    for category, score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {score:.3f}")
    
    # Get top themes (those with score > 0.1)
    themes = [category for category, score in 
              sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
              if score > 0.1]  # Lower threshold for more themes
    
    if not themes:
        return ['miscellaneous']
    
    return themes[:3]  # Return top 3 themes maximum

def cluster_images(reduced_df, output_dir, num_clusters):
    """Perform clustering on the reduced features"""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(reduced_df)
    
    # Add cluster assignments to the dataframe
    clustered_df = pd.DataFrame(index=reduced_df.index)
    clustered_df['cluster'] = clusters
    clustered_df.to_csv(os.path.join(output_dir, 'clustered_data.csv'))
    return clustered_df

def select_representative_images(cluster_data, reduced_features, n_samples=20):
    """Select representative images from a cluster that are at various distances from the center"""
    if len(cluster_data) <= n_samples:
        return list(cluster_data.index), len(cluster_data)
    
    # Get features for this cluster
    cluster_features = reduced_features.loc[cluster_data.index]
    
    # Calculate cluster center
    center = cluster_features.mean(axis=0)
    
    # Calculate distances from center for all points
    distances = np.linalg.norm(cluster_features.values - center.values, axis=1)
    distance_df = pd.DataFrame({'distance': distances}, index=cluster_features.index)
    
    # Sort by distance
    distance_df = distance_df.sort_values('distance')
    
    # Select images at different distance ranges
    n_ranges = 4  # Number of distance ranges to sample from
    samples_per_range = n_samples // n_ranges
    selected_images = []
    
    for i in range(n_ranges):
        start_idx = i * len(distance_df) // n_ranges
        end_idx = (i + 1) * len(distance_df) // n_ranges
        range_df = distance_df.iloc[start_idx:end_idx]
        selected = range_df.sample(min(samples_per_range, len(range_df))).index.tolist()
        selected_images.extend(selected)
    
    # Add random samples if we haven't reached n_samples
    remaining = n_samples - len(selected_images)
    if remaining > 0:
        unselected = list(set(cluster_data.index) - set(selected_images))
        if unselected:
            selected_images.extend(np.random.choice(unselected, min(remaining, len(unselected)), replace=False))
    
    return selected_images, len(cluster_data)

def generate_group_html(df, image_labels, output_dir, quick_view=False):
    """Generate HTML visualization of posts grouped by category"""
    # Group descriptions for the HTML
    GROUP_DESCRIPTIONS = {
        'natural': 'Posts depicting wild fungi in their natural environment with scientific or educational captions.',
        'stylized': 'Posts with artistic filters, soft light, and romanticized or mystical presentation of fungi.',
        'staged': 'Posts showing posed setups, studio shots, or influencer content with mushrooms as props.',
        'symbolic': 'Posts featuring mushrooms as aesthetic symbols, including AI art and fantasy edits.'
    }
    
    template_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fungi Posts by Category</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .group { margin-bottom: 40px; }
            .group-title { 
                font-size: 24px; 
                margin-bottom: 10px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
            .group-description {
                font-size: 16px;
                color: #666;
                margin-bottom: 20px;
                padding: 0 10px;
                font-style: italic;
            }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
                gap: 10px;
            }
            .image-container {
                width: 100%;
                aspect-ratio: 1;
                overflow: visible;
                border-radius: 4px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                position: relative;
            }
            .image-container img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                border-radius: 4px;
                transition: all 0.2s ease;
            }
            .tooltip {
                display: none;
                position: absolute;
                z-index: 100;
                background-color: rgba(0, 0, 0, 0.95);
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
                width: 200px;
                bottom: 110%;
                left: 50%;
                transform: translateX(-50%);
                box-shadow: 0 1px 5px rgba(0,0,0,0.2);
                pointer-events: none;
            }
            .tooltip::after {
                content: '';
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -10px;
                border-width: 10px;
                border-style: solid;
                border-color: rgba(0, 0, 0, 0.95) transparent transparent transparent;
            }
            .image-container:hover .tooltip {
                display: block;
                animation: fadeIn 0.3s ease-in-out forwards;
            }
            .image-container:hover img {
                transform: scale(1.02);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .tooltip-content {
                max-height: 200px;
                overflow-y: auto;
                line-height: 1.4;
            }
            .omitted-count {
                background-color: #f0f0f0;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                font-size: 14px;
                color: #666;
                margin: 10px;
                grid-column: span 2;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translate(-50%, 10px); }
                to { opacity: 1; transform: translate(-50%, 0); }
            }
        </style>
    </head>
    <body>
        {% for group in groups %}
        <div class="group">
            <h2 class="group-title">{{ group.title }} ({{ group.count }} posts)</h2>
            <p class="group-description">{{ group.description }}</p>
            <div class="image-grid">
                {% for image in group.images %}
                <div class="image-container">
                    <img src="https://raw.githubusercontent.com/oraboy/fungi-clustering/main/images/{{ image.path.split('/')[-1] }}" loading="lazy" alt="Image {{ image.path.split('/')[-1] }}">
                    <div class="tooltip">
                        <div class="tooltip-content">
                            <strong>Labels:</strong> {{ image.labels|join(', ') }}
                        </div>
                    </div>
                </div>
                {% endfor %}
                {% if group.omitted > 0 %}
                <div class="omitted-count">+ {{ group.omitted }} more posts</div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
    </body>
    </html>
    """
    
    # Group the posts by category
    groups_data = []
    for category in ['natural', 'stylized', 'staged', 'symbolic']:
        category_posts = df[df['category'] == category]
        
        if quick_view:
            # Select a sample of posts for quick view
            if len(category_posts) > 20:
                category_posts = category_posts.sample(n=20, random_state=42)
        
        images = []
        for _, post in category_posts.iterrows():
            img_name = clean_filename(post['image_url'])
            images.append({
                'path': f'../images/{img_name}',
                'labels': image_labels.get(img_name, [])
            })
        
        groups_data.append({
            'title': category.title(),
            'description': GROUP_DESCRIPTIONS[category],
            'count': len(df[df['category'] == category]),
            'images': images,
            'omitted': len(df[df['category'] == category]) - len(images) if quick_view else 0
        })
    
    # Render template
    template = Template(template_string)
    html_content = template.render(groups=groups_data)
    
    # Write to file
    output_file = os.path.join(output_dir, 'groups_quick.html' if quick_view else 'groups.html')
    with open(output_file, 'w') as f:
        f.write(html_content)

def generate_cluster_html(clustered_df, df, image_labels, reduced_features, output_dir, num_clusters, quick_view=False):
    # Get cluster themes
    cluster_themes = {}
    for cluster_idx in range(num_clusters):
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_idx]
        themes = get_cluster_themes(cluster_data, df, image_labels)
        cluster_themes[cluster_idx] = ' + '.join(themes).title()
    """Generate HTML visualization of clusters"""
    template_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fungi Image Clusters</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .cluster { margin-bottom: 40px; }
            .cluster-title { 
                font-size: 24px; 
                margin-bottom: 10px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
            .cluster-theme {
                font-size: 16px;
                color: #666;
                margin-bottom: 20px;
                padding: 0 10px;
                font-style: italic;
            }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
                gap: 10px;
            }
            .image-container {
                width: 100%;
                aspect-ratio: 1;
                overflow: visible;
                border-radius: 4px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                position: relative;
            }
            .image-container img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                border-radius: 4px;
                transition: all 0.2s ease;
            }
            .tooltip {
                display: none;
                position: absolute;
                z-index: 100;
                background-color: rgba(0, 0, 0, 0.95);
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
                width: 200px;
                bottom: 110%;
                left: 50%;
                transform: translateX(-50%);
                box-shadow: 0 1px 5px rgba(0,0,0,0.2);
                pointer-events: none;
            }
            .tooltip::after {
                content: '';
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -10px;
                border-width: 10px;
                border-style: solid;
                border-color: rgba(0, 0, 0, 0.95) transparent transparent transparent;
            }
            .image-container:hover .tooltip {
                display: block;
                animation: fadeIn 0.3s ease-in-out forwards;
            }
            .image-container:hover img {
                transform: scale(1.02);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .tooltip-content {
                max-height: 200px;
                overflow-y: auto;
                line-height: 1.4;
            }
            .omitted-count {
                background-color: #f0f0f0;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                font-size: 14px;
                color: #666;
                margin: 10px;
                grid-column: span 2;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translate(-50%, 10px); }
                to { opacity: 1; transform: translate(-50%, 0); }
            }
        </style>
    </head>
    <body>
        {% for cluster_num in range(num_clusters) %}
        <div class="cluster">
            <h2 class="cluster-title">Cluster {{ cluster_num + 1 }}</h2>
            <div class="cluster-theme">{{ cluster_themes[cluster_num] }}</div>
            <div class="image-grid">
                {% for image, metadata in clusters[cluster_num] %}
                {% if image != 'omitted' %}
                <div class="image-container">
                    <img src="https://raw.githubusercontent.com/oraboy/fungi-clustering/main/images/{{ image }}" alt="Image {{ image }}">
                    <div class="tooltip">
                        <div class="tooltip-content">
                            <strong>Labels:</strong><br>{{ metadata['labels'] }}<br>
                            <strong>Date:</strong> {{ metadata['date'] }}<br>
                            <strong>Caption:</strong> {{ metadata['caption'] }}
                        </div>
                    </div>
                </div>
                {% else %}
                <div class="omitted-count">
                    + {{ metadata['count'] }} more posts
                </div>
                {% endif %}
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </body>
    </html>
    """
    
    # Organize images by cluster with metadata
    clusters = [[] for _ in range(num_clusters)]
    omitted_counts = [0] * num_clusters
    
    for cluster_idx in range(num_clusters):
        # Get cluster data
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_idx]
        
        # Select images to display
        if quick_view:
            selected_images, total_count = select_representative_images(cluster_data, reduced_features)
            omitted_counts[cluster_idx] = total_count - len(selected_images)
            image_list = selected_images
        else:
            image_list = cluster_data.index
        
        # Process selected images
        for image in image_list:
            # Find original metadata
            original_url = df[df['image_url'].apply(lambda x: clean_filename(x)) == image]['image_url'].iloc[0]
            image_data = df[df['image_url'] == original_url].iloc[0]
            
            # Create metadata dict
            metadata = {
                'date': image_data['post_date'],
                'caption': image_data['post_text'],
                'labels': '\n'.join(image_labels.get(image, []))
            }
            
            clusters[cluster_idx].append((image, metadata))
        
        # Add omitted count if any
        if quick_view and omitted_counts[cluster_idx] > 0:
            clusters[cluster_idx].append(('omitted', {'count': omitted_counts[cluster_idx]}))
    
    # Render template
    template = Template(template_string)
    html_content = template.render(
        clusters=clusters,
        num_clusters=num_clusters,
        cluster_themes=cluster_themes
    )
    
    # Save HTML file
    output_file = 'clusters_quick.html' if quick_view else 'clusters.html'
    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write(html_content)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cluster fungi images using Google Vision API')
    parser.add_argument('--limit', type=int, help='Limit the number of images to process')
    parser.add_argument('--refresh-vision', action='store_true', help='Force refresh of Vision API features')
    parser.add_argument('--samples-per-cluster', type=int, default=20, help='Number of sample images per cluster in quick view')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--num-clusters', type=int, default=DEFAULT_NUM_CLUSTERS, help=f'Number of clusters to generate (default: {DEFAULT_NUM_CLUSTERS})')
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = args.output_dir
    setup_directories(output_dir)
    
    # Read input CSV
    print("\nReading metadata file...")
    df = pd.read_csv(CSV_INPUT_PATH)
    
    if args.limit:
        print(f"\nLimiting to first {args.limit} images for development")
        df = df.head(args.limit)
    
    # Process images and extract features
    features_df, image_labels, image_objects, image_web_entities = process_images(df, refresh_vision=args.refresh_vision)
    
    if len(features_df) == 0:
        print("No features were extracted. Please check the errors above.")
        return
        
    # Filter posts
    df = filter_posts(df, image_labels)
    # Keep only features for non-filtered posts
    features_df = features_df[features_df.index.isin(df['image_url'].apply(clean_filename))]
    
    # Save features and labels to cache if they were newly extracted
    if args.refresh_vision:
        save_features_and_labels(features_df, image_labels, image_objects, image_web_entities)
    
    # Skip group visualizations if category column is missing
    if 'category' in df.columns:
        print("\nGenerating group visualizations...")
        generate_group_html(df, image_labels, output_dir, quick_view=False)
        generate_group_html(df, image_labels, output_dir, quick_view=True)
        
        # Cluster each group separately
        for category in ['natural', 'stylized', 'staged', 'symbolic']:
            category_df = df[df['category'] == category]
            if len(category_df) < 2:  # Skip if not enough posts
                continue
            
            print(f"\nProcessing {category} group...")
            category_features = features_df[features_df.index.isin(category_df['image_url'].apply(clean_filename))]
        
            # Reduce dimensions for this category
            category_reduced = reduce_dimensions(category_features, os.path.join(output_dir, category))
            if category_reduced is None:
                print(f"Dimensionality reduction failed for {category} group. Skipping.")
                continue
            
            # Convert to DataFrame with correct index
            category_reduced_df = pd.DataFrame(category_reduced, index=category_features.index)
            
            # Cluster images in this category
            num_clusters = min(args.num_clusters, len(category_df) // 5)  # Adjust number of clusters based on group size
            if num_clusters < 2:
                continue
            
            print(f"Clustering {category} posts into {num_clusters} clusters...")
            category_clustered = cluster_images(category_reduced_df, os.path.join(output_dir, category), num_clusters)
        
        # Generate cluster visualizations for this category
            print(f"Generating cluster visualizations for {category} group...")
            generate_cluster_html(
                category_clustered, category_df, image_labels, category_reduced_df,
                os.path.join(output_dir, category), num_clusters, quick_view=False
            )
            generate_cluster_html(
                category_clustered, category_df, image_labels, category_reduced_df,
                os.path.join(output_dir, category), num_clusters, quick_view=True
            )
    
    print("\nProcessing complete! Results are in the output directory:")
    print(f"1. Raw features: {os.path.join(output_dir, 'extracted_features.csv')}")
    print(f"2. Image labels: {os.path.join(output_dir, 'image_labels.csv')}")
    print(f"3. Group overview: {os.path.join(output_dir, 'groups.html')}")
    print(f"4. Quick group view: {os.path.join(output_dir, 'groups_quick.html')}")
    for category in ['natural', 'stylized', 'staged', 'symbolic']:
        category_dir = os.path.join(output_dir, category)
        if os.path.exists(category_dir):
            print(f"\n{category.title()} group results:")
            print(f"1. PCA results: {os.path.join(category_dir, 'pca_variance.csv')}")
            print(f"2. Cluster analysis: {os.path.join(category_dir, 'cluster_analysis.csv')}")
            print(f"3. Visual results: {os.path.join(category_dir, 'clusters.html')}")
            print(f"4. Quick view: {os.path.join(category_dir, 'clusters_quick.html')}")

if __name__ == "__main__":
    main()
