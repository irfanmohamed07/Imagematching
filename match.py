"""
Crescent College - Advanced Face Matching System
================================================
Uses multiple advanced ML techniques:
1. MTCNN for face detection with multi-stage cascaded CNNs
2. FaceNet (InceptionResnetV1) for 512-d embedding extraction
3. ArcFace similarity for better angular margin
4. Multi-model ensemble for higher accuracy
5. Face quality assessment for better matching
6. Data augmentation for robustness
7. KNN-based matching with distance metrics
"""

import sys
import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings('ignore')

# ============================================
# DEVICE CONFIGURATION
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", file=sys.stderr)

# ============================================
# MODEL INITIALIZATION
# ============================================

# MTCNN for face detection (multi-task cascaded CNN)
# - Uses 3-stage cascaded architecture: P-Net, R-Net, O-Net
# - Detects faces and facial landmarks simultaneously
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],  # Thresholds for P-Net, R-Net, O-Net
    factor=0.709,
    post_process=True,
    keep_all=False,
    device=device
)

# FaceNet with VGGFace2 pretrained weights
# - Trained on 3.31M images of 9131 subjects
# - Produces 512-dimensional embeddings
resnet_vggface2 = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# FaceNet with CASIA-WebFace pretrained weights (ensemble member)
# - Trained on 500K images of 10K subjects
# - Different training data provides complementary features
resnet_casia = InceptionResnetV1(pretrained='casia-webface').eval().to(device)


# ============================================
# FACE QUALITY ASSESSMENT
# ============================================

def assess_face_quality(face_tensor):
    """
    Assess the quality of detected face for reliable matching
    Returns quality score between 0-1
    
    Factors considered:
    - Brightness: too dark or too bright reduces quality
    - Contrast: low contrast faces are harder to match
    - Blur detection: blurry faces have less reliable features
    - Face size: larger faces have more detail
    """
    if face_tensor is None:
        return 0.0
    
    # Convert to numpy for analysis
    face_np = face_tensor.cpu().numpy()
    if face_np.ndim == 4:
        face_np = face_np[0]
    
    # Transpose from CHW to HWC
    face_np = np.transpose(face_np, (1, 2, 0))
    
    # Normalize to 0-255 range
    face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)
    
    # Convert to grayscale for analysis
    gray = np.mean(face_np, axis=2)
    
    # Brightness score (optimal around 127)
    brightness = np.mean(gray)
    brightness_score = 1 - abs(brightness - 127) / 127
    
    # Contrast score (higher is better)
    contrast = np.std(gray)
    contrast_score = min(contrast / 50, 1.0)
    
    # Sharpness score using Laplacian variance
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    from scipy import ndimage
    try:
        laplacian_var = ndimage.laplace(gray).var()
        sharpness_score = min(laplacian_var / 500, 1.0)
    except:
        sharpness_score = 0.5
    
    # Weighted quality score
    quality = (
        brightness_score * 0.2 +
        contrast_score * 0.3 +
        sharpness_score * 0.5
    )
    
    return float(quality)


# ============================================
# IMAGE PREPROCESSING & AUGMENTATION
# ============================================

def preprocess_image(image):
    """
    Apply preprocessing to improve face detection
    - Auto-contrast enhancement
    - Noise reduction
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # Slight sharpening
    image = image.filter(ImageFilter.SHARPEN)
    
    return image


def apply_test_time_augmentation(image):
    """
    Apply test-time augmentation (TTA) for more robust embeddings
    Returns list of augmented images
    """
    augmented = [image]
    
    # Horizontal flip
    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Slight brightness variations
    enhancer = ImageEnhance.Brightness(image)
    augmented.append(enhancer.enhance(0.9))
    augmented.append(enhancer.enhance(1.1))
    
    # Slight contrast variations
    enhancer = ImageEnhance.Contrast(image)
    augmented.append(enhancer.enhance(0.9))
    augmented.append(enhancer.enhance(1.1))
    
    return augmented


# ============================================
# EMBEDDING EXTRACTION
# ============================================

def get_face_embedding_single(image_path, mtcnn_model, resnet_model, use_augmentation=False):
    """
    Extract face embedding from a single image
    Returns: 512-dimensional normalized embedding or None
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = preprocess_image(img)
        
        if use_augmentation:
            # Test-time augmentation
            augmented_images = apply_test_time_augmentation(img)
            embeddings = []
            
            for aug_img in augmented_images:
                face = mtcnn_model(aug_img)
                if face is not None:
                    if face.dim() == 3:
                        face = face.unsqueeze(0)
                    face = face.to(device)
                    
                    with torch.no_grad():
                        emb = resnet_model(face)
                        emb = F.normalize(emb, p=2, dim=1)
                        embeddings.append(emb.cpu().numpy().flatten())
            
            if embeddings:
                # Average embeddings from augmentations
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                return avg_embedding
            return None
        else:
            # Standard single-image embedding
            face = mtcnn_model(img)
            
            if face is None:
                return None
            
            if face.dim() == 3:
                face = face.unsqueeze(0)
            face = face.to(device)
            
            with torch.no_grad():
                embedding = resnet_model(face)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}", file=sys.stderr)
        return None


def get_ensemble_embedding(image_path, use_augmentation=True):
    """
    Get ensemble embedding using multiple models
    Combines VGGFace2 and CASIA-WebFace pretrained models
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = preprocess_image(img)
        
        # Detect face once
        face = mtcnn(img)
        
        if face is None:
            return None, None, 0.0
        
        # Assess face quality
        quality_score = assess_face_quality(face)
        
        if face.dim() == 3:
            face = face.unsqueeze(0)
        face = face.to(device)
        
        # Get embeddings from both models
        with torch.no_grad():
            emb_vgg = resnet_vggface2(face)
            emb_casia = resnet_casia(face)
            
            # L2 normalize
            emb_vgg = F.normalize(emb_vgg, p=2, dim=1)
            emb_casia = F.normalize(emb_casia, p=2, dim=1)
        
        emb_vgg = emb_vgg.cpu().numpy().flatten()
        emb_casia = emb_casia.cpu().numpy().flatten()
        
        # Weighted ensemble (VGGFace2 slightly weighted more due to larger training set)
        ensemble_emb = 0.6 * emb_vgg + 0.4 * emb_casia
        ensemble_emb = ensemble_emb / np.linalg.norm(ensemble_emb)
        
        return ensemble_emb, emb_vgg, quality_score
    
    except Exception as e:
        print(f"Error in ensemble embedding: {str(e)}", file=sys.stderr)
        return None, None, 0.0


# ============================================
# SIMILARITY METRICS
# ============================================

def cosine_similarity(emb1, emb2):
    """Standard cosine similarity"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def arcface_similarity(emb1, emb2, s=30.0, m=0.50):
    """
    ArcFace-inspired angular similarity
    - Uses geodesic distance in embedding space
    - Better separates similar identities
    
    s: scale factor
    m: angular margin (not applied during inference, just for reference)
    """
    cos_theta = cosine_similarity(emb1, emb2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Convert to angular distance
    theta = np.arccos(cos_theta)
    
    # Convert back to similarity (normalized to 0-1)
    arc_sim = 1 - (theta / np.pi)
    
    return float(arc_sim)


def euclidean_distance(emb1, emb2):
    """Euclidean distance between embeddings"""
    return np.linalg.norm(emb1 - emb2)


def combined_similarity(emb1, emb2):
    """
    Combined similarity using multiple metrics
    Weighted average of cosine and ArcFace similarities
    """
    cos_sim = cosine_similarity(emb1, emb2)
    arc_sim = arcface_similarity(emb1, emb2)
    
    # Weighted combination
    combined = 0.5 * cos_sim + 0.5 * arc_sim
    
    return float(combined)


# ============================================
# EMBEDDING STORAGE & KNN MATCHING
# ============================================

def load_image_metadata(metadata_file="image_data.json"):
    """Load image metadata from JSON file"""
    try:
        with open(metadata_file, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading metadata: {str(e)}", file=sys.stderr)
        return {}


def load_stored_embeddings(images_dir="images", metadata_file="image_data.json", force_regenerate=False):
    """
    Load or generate embeddings for all stored images
    Uses ensemble embeddings for better accuracy
    """
    embeddings_cache_file = "embeddings_cache.json"
    embeddings = {}
    quality_scores = {}
    
    # Try to load cached embeddings
    if os.path.exists(embeddings_cache_file) and not force_regenerate:
        try:
            with open(embeddings_cache_file, "r") as f:
                cached_data = json.load(f)
                for filename, data in cached_data.items():
                    if isinstance(data, dict):
                        embeddings[filename] = np.array(data.get("embedding", data))
                        quality_scores[filename] = data.get("quality", 0.8)
                    else:
                        embeddings[filename] = np.array(data)
                        quality_scores[filename] = 0.8
        except Exception as e:
            print(f"Cache load error: {str(e)}", file=sys.stderr)
    
    # Generate embeddings for missing images
    image_metadata = load_image_metadata(metadata_file)
    needs_update = False
    
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                if filename in image_metadata and filename not in embeddings:
                    image_path = os.path.join(images_dir, filename)
                    
                    # Use ensemble embedding for stored images
                    embedding, _, quality = get_ensemble_embedding(image_path)
                    
                    if embedding is not None:
                        embeddings[filename] = embedding
                        quality_scores[filename] = quality
                        needs_update = True
                        print(f"Generated embedding for {filename} (quality: {quality:.2f})", file=sys.stderr)
    
    # Save updated embeddings cache with quality scores
    if needs_update:
        cache_data = {
            filename: {
                "embedding": emb.tolist(),
                "quality": quality_scores.get(filename, 0.8)
            }
            for filename, emb in embeddings.items()
        }
        with open(embeddings_cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    
    return embeddings, quality_scores


class FaceMatcherKNN:
    """
    KNN-based face matcher for efficient retrieval
    Uses approximate nearest neighbors for scalability
    """
    
    def __init__(self, embeddings_dict, n_neighbors=5):
        self.filenames = list(embeddings_dict.keys())
        self.embeddings = np.array([embeddings_dict[f] for f in self.filenames])
        
        # Normalize embeddings
        self.embeddings = normalize(self.embeddings, norm='l2')
        
        # Initialize KNN with cosine similarity (via angular distance)
        self.knn = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(self.filenames)),
            metric='cosine',
            algorithm='brute'  # Use 'ball_tree' for larger datasets
        )
        self.knn.fit(self.embeddings)
    
    def find_matches(self, query_embedding, threshold=0.6):
        """
        Find matching faces using KNN
        Returns list of (filename, similarity) tuples
        """
        query_emb = query_embedding.reshape(1, -1)
        query_emb = normalize(query_emb, norm='l2')
        
        distances, indices = self.knn.kneighbors(query_emb)
        
        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert cosine distance to similarity
            similarity = 1 - dist
            if similarity >= threshold:
                matches.append((self.filenames[idx], float(similarity)))
        
        return matches


# ============================================
# MAIN MATCHING FUNCTION
# ============================================

def match_image(uploaded_image_path, predefined_images_dir="images",
                metadata_file="image_data.json", confidence_threshold=0.55):
    """
    Advanced face matching using ensemble embeddings and multiple similarity metrics
    
    Pipeline:
    1. Load stored embeddings (with caching)
    2. Extract ensemble embedding from uploaded image
    3. Assess face quality
    4. Use KNN for initial candidate retrieval
    5. Re-rank using combined similarity metric
    6. Return top matches with confidence scores
    """
    try:
        # Load stored embeddings
        stored_embeddings, stored_quality = load_stored_embeddings(
            predefined_images_dir, metadata_file
        )
        
        if not stored_embeddings:
            result = {
                "matches": [],
                "error": "No stored embeddings found. Please add student images first."
            }
            print(json.dumps(result))
            return
        
        # Get ensemble embedding for uploaded image
        uploaded_embedding, vgg_embedding, quality_score = get_ensemble_embedding(uploaded_image_path)
        
        if uploaded_embedding is None:
            result = {
                "matches": [],
                "error": "No face detected in uploaded image. Please use a clear front-facing photo."
            }
            print(json.dumps(result))
            return
        
        # Quality check
        if quality_score < 0.3:
            result = {
                "matches": [],
                "error": f"Image quality too low ({quality_score:.1%}). Please use a clearer photo.",
                "quality_score": round(quality_score, 3)
            }
            print(json.dumps(result))
            return
        
        # Load metadata
        image_metadata = load_image_metadata(metadata_file)
        
        # Method 1: KNN-based matching for efficiency
        matcher = FaceMatcherKNN(stored_embeddings, n_neighbors=5)
        knn_matches = matcher.find_matches(uploaded_embedding, threshold=confidence_threshold)
        
        # Method 2: Re-rank using combined similarity
        matches = []
        for filename, knn_similarity in knn_matches:
            stored_emb = stored_embeddings[filename]
            
            # Calculate multiple similarity metrics
            cos_sim = cosine_similarity(uploaded_embedding, stored_emb)
            arc_sim = arcface_similarity(uploaded_embedding, stored_emb)
            combined_sim = combined_similarity(uploaded_embedding, stored_emb)
            
            # Final confidence (weighted by stored image quality)
            stored_qual = stored_quality.get(filename, 0.8)
            quality_factor = (quality_score + stored_qual) / 2
            
            # Adjust confidence based on quality
            final_confidence = combined_sim * (0.7 + 0.3 * quality_factor)
            
            if final_confidence >= confidence_threshold:
                metadata = image_metadata.get(filename, {})
                matches.append({
                    "name": metadata.get("Name", filename.split('.')[0]),
                    "roll_no": metadata.get("RRN", "N/A"),
                    "department": metadata.get("Department", "N/A"),
                    "year": metadata.get("Year", "N/A"),
                    "section": metadata.get("Section", "N/A"),
                    "confidence": round(final_confidence, 4),
                    "cosine_sim": round(cos_sim, 4),
                    "arcface_sim": round(arc_sim, 4),
                    "filename": filename,
                    "quality_factor": round(quality_factor, 3)
                })
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Prepare result
        result = {
            "matches": matches[:5],  # Top 5 matches
            "total_matches": len(matches),
            "uploaded_image_quality": round(quality_score, 3),
            "model_info": {
                "detection": "MTCNN (Multi-task Cascaded CNN)",
                "embedding": "FaceNet Ensemble (VGGFace2 + CASIA-WebFace)",
                "similarity": "Combined (Cosine + ArcFace)",
                "matching": "KNN with re-ranking"
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        result = {
            "matches": [],
            "error": f"Processing error: {str(e)}"
        }
        print(json.dumps(result))


# ============================================
# CLI ENTRY POINT
# ============================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
    
    uploaded_image_path = sys.argv[1]
    
    # Optional: force regenerate embeddings cache
    if len(sys.argv) > 2 and sys.argv[2] == "--regenerate":
        print("Regenerating embeddings cache...", file=sys.stderr)
        load_stored_embeddings(force_regenerate=True)
    
    match_image(uploaded_image_path)
