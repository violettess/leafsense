import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rembg import remove
import io

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoint.pth"  # Place your checkpoint.pth in the same directory
IMG_SIZE = 380

# 38 PlantVillage Classes (alphabetically ordered as per ImageFolder)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease Information Database
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'name': 'Apple Scab',
        'pathogen': 'Venturia inaequalis (fungus)',
        'symptoms': 'Olive-green to brown lesions on leaves and fruit. Leaves may curl and drop early.',
        'management': 'Remove fallen leaves, apply fungicides during wet periods, use resistant varieties.'
    },
    'Apple___Black_rot': {
        'name': 'Apple Black Rot',
        'pathogen': 'Botryosphaeria obtusa (fungus)',
        'symptoms': 'Brown circular lesions with concentric rings on fruit. Leaf spots with purple margins.',
        'management': 'Prune dead wood, remove mummified fruit, apply fungicides in spring.'
    },
    'Apple___Cedar_apple_rust': {
        'name': 'Cedar Apple Rust',
        'pathogen': 'Gymnosporangium juniperi-virginianae (fungus)',
        'symptoms': 'Yellow-orange spots on leaves, fruit may develop raised lesions.',
        'management': 'Remove nearby cedar trees, apply fungicides before symptoms appear.'
    },
    'Apple___healthy': {
        'name': 'Healthy Apple',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Maintain good cultural practices and regular monitoring.'
    },
    'Blueberry___healthy': {
        'name': 'Healthy Blueberry',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Maintain proper pH (4.5-5.5), ensure good drainage and mulching.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'name': 'Cherry Powdery Mildew',
        'pathogen': 'Podosphaera clandestina (fungus)',
        'symptoms': 'White powdery coating on leaves, shoots, and fruit. Leaves may curl upward.',
        'management': 'Improve air circulation, apply sulfur or fungicides, remove infected tissue.'
    },
    'Cherry_(including_sour)___healthy': {
        'name': 'Healthy Cherry',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Prune for good airflow, monitor regularly for pests and diseases.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'name': 'Gray Leaf Spot',
        'pathogen': 'Cercospora zeae-maydis (fungus)',
        'symptoms': 'Rectangular gray-brown lesions between leaf veins. Severe infection causes premature leaf death.',
        'management': 'Crop rotation, use resistant hybrids, apply fungicides if necessary.'
    },
    'Corn_(maize)___Common_rust_': {
        'name': 'Common Rust',
        'pathogen': 'Puccinia sorghi (fungus)',
        'symptoms': 'Circular to elongated reddish-brown pustules on both leaf surfaces.',
        'management': 'Plant resistant varieties, apply fungicides during early infection.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'name': 'Northern Leaf Blight',
        'pathogen': 'Exserohilum turcicum (fungus)',
        'symptoms': 'Long grayish-green or tan cigar-shaped lesions on leaves.',
        'management': 'Crop rotation, resistant hybrids, fungicide application in severe cases.'
    },
    'Corn_(maize)___healthy': {
        'name': 'Healthy Corn',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Balanced fertilization, proper spacing, and timely irrigation.'
    },
    'Grape___Black_rot': {
        'name': 'Grape Black Rot',
        'pathogen': 'Guignardia bidwellii (fungus)',
        'symptoms': 'Circular tan spots with dark borders on leaves. Fruit turns black and shrivels.',
        'management': 'Remove mummified berries, apply fungicides from bloom to fruit set.'
    },
    'Grape___Esca_(Black_Measles)': {
        'name': 'Esca (Black Measles)',
        'pathogen': 'Complex of fungi including Phaeomoniella and Phaeoacremonium',
        'symptoms': 'Tiger-stripe pattern on leaves, black spots on berries, vine decline.',
        'management': 'No cure; remove infected vines, protect pruning wounds, delay pruning.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'name': 'Isariopsis Leaf Spot',
        'pathogen': 'Pseudocercospora vitis (fungus)',
        'symptoms': 'Dark brown to black angular spots on leaves, premature defoliation.',
        'management': 'Improve canopy ventilation, apply copper-based fungicides.'
    },
    'Grape___healthy': {
        'name': 'Healthy Grape',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Proper pruning, canopy management, and regular scouting.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'name': 'Huanglongbing (Citrus Greening)',
        'pathogen': 'Candidatus Liberibacter (bacterium)',
        'symptoms': 'Yellow shoots, blotchy mottled leaves, lopsided bitter fruit.',
        'management': 'Remove infected trees, control psyllid vectors, use disease-free nursery stock.'
    },
    'Peach___Bacterial_spot': {
        'name': 'Bacterial Spot',
        'pathogen': 'Xanthomonas arboricola pv. pruni (bacterium)',
        'symptoms': 'Small dark spots on leaves with yellow halos. Fruit lesions are raised and cracked.',
        'management': 'Copper sprays, resistant varieties, reduce overhead irrigation.'
    },
    'Peach___healthy': {
        'name': 'Healthy Peach',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Proper thinning, adequate fertilization, and pest monitoring.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'name': 'Bacterial Spot',
        'pathogen': 'Xanthomonas spp. (bacterium)',
        'symptoms': 'Small water-soaked spots on leaves that turn brown. Fruit spots are raised and scabby.',
        'management': 'Use disease-free seed, copper sprays, crop rotation.'
    },
    'Pepper,_bell___healthy': {
        'name': 'Healthy Bell Pepper',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Consistent watering, mulching, and proper plant spacing.'
    },
    'Potato___Early_blight': {
        'name': 'Early Blight',
        'pathogen': 'Alternaria solani (fungus)',
        'symptoms': 'Circular brown spots with concentric rings (target pattern) on older leaves.',
        'management': 'Crop rotation, fungicide applications, remove infected plant debris.'
    },
    'Potato___Late_blight': {
        'name': 'Late Blight',
        'pathogen': 'Phytophthora infestans (oomycete)',
        'symptoms': 'Water-soaked lesions on leaves that turn brown and black. White fungal growth on undersides.',
        'management': 'Destroy infected plants, preventive fungicides, avoid overhead irrigation.'
    },
    'Potato___healthy': {
        'name': 'Healthy Potato',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Use certified seed potatoes, hill soil properly, monitor for pests.'
    },
    'Raspberry___healthy': {
        'name': 'Healthy Raspberry',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Prune out old canes, ensure good drainage, mulch well.'
    },
    'Soybean___healthy': {
        'name': 'Healthy Soybean',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Crop rotation, proper seeding rate, monitor for pests.'
    },
    'Squash___Powdery_mildew': {
        'name': 'Powdery Mildew',
        'pathogen': 'Podosphaera xanthii (fungus)',
        'symptoms': 'White powdery coating on leaves and stems. Leaves turn yellow and die.',
        'management': 'Sulfur or fungicides, improve air circulation, resistant varieties.'
    },
    'Strawberry___Leaf_scorch': {
        'name': 'Leaf Scorch',
        'pathogen': 'Diplocarpon earlianum (fungus)',
        'symptoms': 'Purple spots on leaves that enlarge and turn brown/gray in center.',
        'management': 'Remove infected leaves, fungicide applications, plant resistant varieties.'
    },
    'Strawberry___healthy': {
        'name': 'Healthy Strawberry',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Mulch properly, ensure adequate spacing, renovate beds annually.'
    },
    'Tomato___Bacterial_spot': {
        'name': 'Bacterial Spot',
        'pathogen': 'Xanthomonas spp. (bacterium)',
        'symptoms': 'Small dark brown spots with yellow halos on leaves. Raised spots on fruit.',
        'management': 'Use disease-free transplants, copper bactericides, avoid overhead watering.'
    },
    'Tomato___Early_blight': {
        'name': 'Early Blight',
        'pathogen': 'Alternaria solani (fungus)',
        'symptoms': 'Brown spots with concentric rings on lower leaves. Lesions on stems and fruit.',
        'management': 'Stake plants, mulch, fungicide applications, crop rotation.'
    },
    'Tomato___Late_blight': {
        'name': 'Late Blight',
        'pathogen': 'Phytophthora infestans (oomycete)',
        'symptoms': 'Large brown-black blotches on leaves and stems. White fungal growth in humid conditions.',
        'management': 'Remove infected plants immediately, preventive fungicides, improve air circulation.'
    },
    'Tomato___Leaf_Mold': {
        'name': 'Leaf Mold',
        'pathogen': 'Passalora fulva (fungus)',
        'symptoms': 'Yellow spots on upper leaf surface, olive-green to gray mold on undersides.',
        'management': 'Reduce humidity, increase ventilation, apply fungicides.'
    },
    'Tomato___Septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'pathogen': 'Septoria lycopersici (fungus)',
        'symptoms': 'Small circular spots with dark borders and gray centers containing black specks.',
        'management': 'Remove infected leaves, mulch soil, fungicide sprays.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'name': 'Spider Mites',
        'pathogen': 'Tetranychus urticae (arachnid pest)',
        'symptoms': 'Stippling on leaves, fine webbing, yellowing and bronzing of foliage.',
        'management': 'Use miticides, spray with water, introduce predatory mites.'
    },
    'Tomato___Target_Spot': {
        'name': 'Target Spot',
        'pathogen': 'Corynespora cassiicola (fungus)',
        'symptoms': 'Brown spots with concentric rings on leaves and fruit.',
        'management': 'Fungicide applications, crop rotation, remove plant debris.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Yellow Leaf Curl Virus',
        'pathogen': 'Begomovirus (virus)',
        'symptoms': 'Upward curling and yellowing of leaves, stunted growth, reduced fruit production.',
        'management': 'Control whitefly vectors, use virus-resistant varieties, remove infected plants.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus',
        'pathogen': 'Tobamovirus (virus)',
        'symptoms': 'Mottled light and dark green pattern on leaves, distorted leaves, stunted plants.',
        'management': 'Use resistant varieties, sanitize tools, remove infected plants.'
    },
    'Tomato___healthy': {
        'name': 'Healthy Tomato',
        'pathogen': 'None',
        'symptoms': 'No visible symptoms of disease.',
        'management': 'Proper staking, consistent watering, balanced fertilization.'
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load EfficientNet-B4 model with trained weights"""
    model = models.efficientnet_b4(weights=None)
    
    # Replace classifier to match 38 classes
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 38)
    )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    model = model.to(DEVICE)
    model.eval()
    return model

# ═══════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def remove_background(image):
    """Remove background using rembg"""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        output = remove(img_byte_arr)
        result = Image.open(io.BytesIO(output)).convert("RGB")
        return result
    except Exception as e:
        st.warning(f"Background removal failed: {e}. Using original image.")
        return image

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict(model, image_tensor):
    """Get top-5 predictions"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)
    
    predictions = []
    for i in range(5):
        predictions.append({
            'class': CLASS_NAMES[top5_idx[0][i].item()],
            'confidence': top5_prob[0][i].item() * 100
        })
    
    return predictions

# ═══════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ═══════════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        self.hook_handles.append(
            target_layer.register_forward_hook(self.save_activation)
        )
        self.hook_handles.append(
            target_layer.register_full_backward_hook(self.save_gradient)
        )
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, index=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if index is None:
            index = output.argmax(dim=1).item()
        
        loss = output[0, index]
        loss.backward()
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        
        grad_cam = F.interpolate(
            grad_cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        grad_cam = grad_cam.squeeze().cpu().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        
        return grad_cam
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def generate_gradcam(model, image, image_tensor, pred_class_idx):
    """Generate GradCAM heatmap overlay"""
    target_layer = model.features[7]
    gradcam = GradCAM(model, target_layer)
    
    heatmap = gradcam(image_tensor, index=pred_class_idx)
    gradcam.remove_hooks()
    
    # Resize image to match heatmap
    img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    
    # Resize heatmap and apply colormap
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = 0.5 * heatmap_color + 0.5 * img_np
    overlay = np.uint8(overlay)
    
    return overlay

# ═══════════════════════════════════════════════════════════════════════════
# DISEASE INFORMATION
# ═══════════════════════════════════════════════════════════════════════════

def get_disease_info(class_name):
    """Get disease information from database"""
    return DISEASE_INFO.get(class_name, {
        'name': 'Unknown',
        'pathogen': 'Information not available',
        'symptoms': 'Information not available',
        'management': 'Information not available'
    })

# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="LeafSense",
        page_icon="🌿",
        layout="wide"
    )
    
    # Title
    st.title("🌿 LeafSense")
    st.markdown("**Plant Disease Detector using EfficientNet-B4 + GradCAM Visualization**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹAbout")
        st.markdown("""
        This application uses **EfficientNet-B4** deep learning model 
        to classify plant leaf diseases across 38 different classes 
        from the PlantVillage dataset.
        
        ### Features:
        - **Background Removal**: Isolates leaf from background
        - **Top-5 Predictions**: Shows confidence scores
        - **GradCAM Visualization**: Highlights diseased regions
        - **Disease Information**: Detailed pathogen & management info
        
        ### Instructions:
        1. Upload a leaf image (JPG/PNG)
        2. Toggle background removal if needed
        3. View predictions and GradCAM
        4. Read disease information
        
        ### Model Details:
        - **Architecture**: EfficientNet-B4
        - **Input Size**: 380×380 pixels
        - **Classes**: 38 disease categories
        - **Device**: {0}
        """.format("GPU (CUDA)" if torch.cuda.is_available() else "CPU"))
        
        st.markdown("---")
        
        # Settings
        st.header("Settings")
        use_bg_removal = st.checkbox("Use Background Removal", value=True, 
                                     help="Uncheck if background removal removes too much of the leaf")
        confidence_threshold = st.slider("Confidence Threshold", 
                                        min_value=50, max_value=95, value=70,
                                        help="Minimum confidence to show specific disease prediction")
        
        st.markdown("---")
        st.markdown("Built with using Streamlit")
    
    # Load model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf"
    )
    
    if uploaded_file is not None:
        # Load image
        original_image = Image.open(uploaded_file).convert("RGB")
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)
        
        # Background removal
        with st.spinner("Removing background..."):
            bg_removed_image = remove_background(original_image)
        
        with col2:
            if use_bg_removal:
                st.subheader("Background Removed")
                st.image(bg_removed_image, use_container_width=True)
            else:
                st.subheader("Original Image (Used for Analysis)")
                st.image(original_image, use_container_width=True)
                st.info("Background removal is disabled. Using original image.")
        
        st.markdown("---")
        
        # Preprocessing and prediction
        with st.spinner("Analyzing leaf..."):
            # Use background removed or original based on setting
            processed_image = bg_removed_image if use_bg_removal else original_image
            image_tensor = preprocess_image(processed_image)
            predictions = predict(model, image_tensor)
            pred_class_idx = CLASS_NAMES.index(predictions[0]['class'])
            top_confidence = predictions[0]['confidence']
        
        # Check if confidence is too low
        if top_confidence < confidence_threshold:
            st.warning(f"⚠️ Low Confidence Detection ({top_confidence:.2f}%)")
            st.markdown("""
            ### 🔍 Analysis Results:
            
            The model's confidence is below {0}%, which suggests this leaf may not belong 
            to any of the 38 trained disease categories.
            
            **Possible reasons:**
            - This is a different plant species not in the training data
            - The disease is not one of the 38 classes the model was trained on
            - Image quality or angle makes classification difficult
            - The leaf might be in early disease stages
            
            ### Visual Assessment:
            """.format(confidence_threshold))
            
            # Simple heuristic for healthy vs unhealthy
            # Check if any "healthy" class is in top 5
            has_healthy_in_top5 = any('healthy' in pred['class'].lower() for pred in predictions)
            
            if has_healthy_in_top5:
                st.success("""
                ✅ **Preliminary Assessment: Likely Healthy**
                
                The leaf appears to have characteristics of a healthy plant based on visual patterns. 
                However, this is not a definitive diagnosis.
                """)
            else:
                st.error("""
                ⚠️ **Preliminary Assessment: Possibly Unhealthy**
                
                The leaf shows visual patterns that may indicate stress or disease, but the specific 
                condition cannot be determined from the trained categories.
                """)
            
            st.info("""
            **Recommendations:**
            - Consult with a local agricultural extension office
            - Take multiple photos from different angles
            - Compare with known disease images for your plant species
            - Monitor the plant for symptom progression
            """)
            
            # Still show predictions but with disclaimer
            st.markdown("---")
            st.subheader("Top Predictions (for reference only)")
            st.caption("⚠️ These predictions have low confidence and should not be relied upon.")
        else:
            # High confidence - show normal prediction UI
            st.subheader("Top 5 Predictions")
        
        for i, pred in enumerate(predictions):
            class_display = pred['class'].replace('___', ' - ').replace('_', ' ')
            confidence = pred['confidence']
            
            # Color coding
            if i == 0:
                color = "🥇"
            elif i == 1:
                color = "🥈"
            elif i == 2:
                color = "🥉"
            else:
                color = f"{i+1}."
            
            st.markdown(f"{color} **{class_display}** - {confidence:.2f}%")
            st.progress(confidence / 100)
        
        st.markdown("---")
        
        # GradCAM visualization
        st.subheader("GradCAM Visualization")
        
        with st.spinner("Generating heatmaps..."):
            overlay_original = generate_gradcam(model, original_image, 
                                               preprocess_image(original_image), 
                                               pred_class_idx)
            
            if use_bg_removal:
                overlay_processed = generate_gradcam(model, bg_removed_image, 
                                                     image_tensor, 
                                                     pred_class_idx)
            else:
                overlay_processed = overlay_original
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image + GradCAM**")
            st.image(overlay_original, use_container_width=True)
        
        with col2:
            if use_bg_removal:
                st.markdown("**Background Removed + GradCAM**")
            else:
                st.markdown("**Processed Image + GradCAM**")
            st.image(overlay_processed, use_container_width=True)
        
        st.info("🔍 Red/yellow regions indicate areas the model focused on for its prediction.")
        
        st.markdown("---")
        
        # Disease information (only show for high confidence)
        if top_confidence >= confidence_threshold:
            st.subheader("Disease Information")
            
            disease_info = get_disease_info(predictions[0]['class'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Disease Name:**")
                st.write(disease_info['name'])
                
                st.markdown(f"**Pathogen:**")
                st.write(disease_info['pathogen'])
            
            with col2:
                st.markdown(f"**Symptoms:**")
                st.write(disease_info['symptoms'])
                
                st.markdown(f"**Management:**")
                st.write(disease_info['management'])
            
            # Warning for healthy leaves
            if 'healthy' in predictions[0]['class'].lower():
                st.success("✅ This leaf appears to be healthy! Continue monitoring for early signs of disease.")
            else:
                st.warning("⚠️ Disease detected. Consult with a local agricultural extension for proper treatment.")
        else:
            st.markdown("---")
            st.info("💡 **Tip:** Try adjusting the confidence threshold in the sidebar or upload a clearer image for better results.")

if __name__ == "__main__":
    main()