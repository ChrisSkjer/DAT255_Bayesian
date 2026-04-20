import os
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# Configuration - From notebook 4
# ============================================================================
@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for the app."""
    image_size: tuple[int, int] = (128, 128)
    mc_samples: int = 30
    # W&B artifact paths
    wandb_entity: str = "christoffer-skjer-h-gskulen-p-vestlandet"
    wandb_project: str = "dat255-bayesian"
    wandb_artifacts: dict[str, str] = field(default_factory=lambda: {
        "Dropout 0.1": "notebook-04-dropout-0-1-model:v0",
        "Dropout 0.3": "notebook-04-dropout-0-3-model:v0",
        "Dropout 0.5": "notebook-04-dropout-0-5-model:v0",
    })


def get_config() -> ProjectConfig:
    """Return the app configuration."""
    return ProjectConfig()


# ============================================================================
# MC Dropout Layer - From notebook 4
# ============================================================================
class MCDropout(tf.keras.layers.Dropout):
    """Dropout layer that stays active during inference for MC sampling."""

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


# ============================================================================
# Uncertainty Utilities - From notebook 4
# ============================================================================
def mc_predict(
    model: tf.keras.Model,
    images: tf.Tensor,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run several stochastic forward passes and summarize the predictions."""
    predictions = []
    for _ in range(mc_samples):
        predictions.append(model(images, training=False).numpy())

    stacked = np.stack(predictions, axis=0)
    mean_prediction = stacked.mean(axis=0)
    variance = stacked.var(axis=0)
    predictive_entropy = -np.sum(
        mean_prediction * np.log(mean_prediction + 1e-10),
        axis=1,
    )
    return mean_prediction, variance, predictive_entropy


# ============================================================================
# Streamlit App Configuration
# ============================================================================
st.set_page_config(
    page_title="ImageNette Classifier with Uncertainty",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🤖 ImageNette Classifier with MC Dropout Uncertainty")
st.markdown(
    """
    This app uses a CNN with Monte Carlo Dropout to classify images from ImageNette
    and estimate prediction uncertainty. Upload an image to get started!
    """
)

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
config = get_config()

# Model selection
st.sidebar.subheader("🤖 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a model by dropout rate:",
    options=list(config.wandb_artifacts.keys()),
    help="Different dropout rates affect uncertainty estimation"
)

# Initialize session state for mc_samples if not exists
if "mc_samples" not in st.session_state:
    st.session_state.mc_samples = config.mc_samples

mc_samples = st.sidebar.slider(
    "MC Samples for Uncertainty Estimation",
    min_value=10,
    max_value=100,
    value=st.session_state.mc_samples,
    step=10,
    help="More samples = more accurate uncertainty, but slower predictions",
    key="mc_samples",
)

# ImageNette class names
CLASS_NAMES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]

@st.cache_resource
def load_model(model_name: str = "Dropout 0.3"):
    """Load a trained model from Weights & Biases."""
    if not HAS_WANDB:
        st.error("❌ wandb package not installed. Install it with: pip install wandb")
        st.stop()
    
    try:
        config = get_config()
        artifact_name = config.wandb_artifacts.get(model_name, config.wandb_artifacts["Dropout 0.3"])
        
        st.info(f"📥 Downloading {model_name} from Weights & Biases...")
        
        # Use read-only artifact access
        api = wandb.Api()
        artifact_path = f"{config.wandb_entity}/{config.wandb_project}/{artifact_name}"
        artifact = api.artifact(artifact_path)
        artifact_dir = artifact.download()
        
        st.success(f"✅ {model_name} downloaded successfully!")
        
        # Find the model file in the artifact directory
        model_files = list(Path(artifact_dir).glob("*.keras"))
        if not model_files:
            st.error(f"❌ No .keras file found in artifact directory: {artifact_dir}")
            st.stop()
        
        model_path = model_files[0]
        
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"MCDropout": MCDropout},
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading {model_name} from W&B: {e}")
        st.stop()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to match model input requirements."""
    # Resize to model input size
    image = image.convert("RGB")
    image = image.resize(config.image_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def main():
    """Main app logic."""
    # Load the selected model
    model = load_model(selected_model)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Upload Image", "📋 About Model"])
    
    with tab1:
        # Image upload section
        st.subheader("Upload an Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to classify",
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Input Image")
                st.image(image, use_column_width=True)
            
            # Make prediction
            with col2:
                st.subheader("🔮 Prediction Results")
                
                with st.spinner("Running MC prediction... this may take a moment"):
                    # Preprocess image
                    img_array = preprocess_image(image)
                    
                    # Run MC predictions
                    mean_prediction, variance, predictive_entropy = mc_predict(
                        model,
                        img_array,
                        mc_samples,
                    )
                
                # Get prediction results for single image
                pred_probs = mean_prediction[0]
                pred_class = np.argmax(pred_probs)
                pred_confidence = pred_probs[pred_class]
                entropy = predictive_entropy[0]
                avg_variance = np.mean(variance[0])
                avg_std = np.sqrt(avg_variance)  # Standard deviation
                
                # Display main prediction
                st.markdown("### 🎯 Top Prediction")
                st.metric(
                    f"Class: {CLASS_NAMES[pred_class]}",
                    f"{pred_confidence * 100:.1f}%",
                    help=f"Class index: {pred_class}"
                )
                
                # Display uncertainty metrics
                st.markdown("### 📊 Uncertainty Metrics")
                col_unc1, col_unc2, col_unc3 = st.columns(3)
                
                with col_unc1:
                    st.metric(
                        "Predictive Entropy",
                        f"{entropy:.4f}",
                        help="Higher entropy = more uncertain prediction",
                    )
                
                with col_unc2:
                    st.metric(
                        "Mean Variance",
                        f"{avg_variance:.6f}",
                        help="Variance across MC samples",
                    )
                
                with col_unc3:
                    st.metric(
                        "Mean Std Dev",
                        f"{avg_std * 100:.4f}%",
                        help="Standard deviation across MC samples",
                    )
            
            # Show top-5 predictions
            st.subheader("📈 Top-5 Predictions")
            top_5_idx = np.argsort(pred_probs)[-5:][::-1]
            
            fig_data = {
                "Class": [CLASS_NAMES[idx] for idx in top_5_idx],
                "Confidence": [pred_probs[idx] * 100 for idx in top_5_idx],
            }
            
            st.bar_chart(
                data={fig_data["Class"][i]: fig_data["Confidence"][i] 
                      for i in range(len(fig_data["Class"]))},
                height=300,
            )
            
            # Detailed probabilities table
            st.markdown("### 📋 All Class Probabilities")
            results_df = []
            for idx, prob in enumerate(pred_probs):
                results_df.append({
                    "Class": CLASS_NAMES[idx],
                    "Probability (%)": f"{prob * 100:.2f}%",
                    "Log Probability": f"{np.log(prob + 1e-10):.4f}",
                })
            
            st.dataframe(results_df, use_container_width=True)
    
    with tab2:
        st.subheader("📚 About This Model")
        
        st.markdown("""
        ### Model Architecture
        This classifier uses a **CNN with MC Dropout** for both classification and uncertainty estimation.
        
        **Key Features:**
        - 🔄 Convolutional layers with MaxPooling
        - 🎲 MC Dropout layers (active at inference for uncertainty)
        - 📊 Outputs class probabilities and uncertainty metrics
        
        ### Dataset
        - **Dataset:** ImageNette (10-class subset of ImageNet)
        - **Image Size:** 128 × 128 pixels
        - **Classes:** tench, springer, cassette player, chain saw, church, 
                      french horn, garbage truck, gas pump, golf ball, parachute
        
        ### Uncertainty Estimation
        - **Method:** Monte Carlo Dropout sampling
        - **Samples:** 30 forward passes (configurable)
        - **Metrics:**
          - **Predictive Entropy:** Uncertainty in class predictions
          - **Variance:** Variability across MC samples
        
        ### How to Interpret Results
        - ✅ **Low entropy** + **High confidence** = Model is confident
        - ⚠️ **High entropy** = Model is uncertain, may be out-of-distribution
        - 📊 Variance shows consistency across MC samples
        """)
        
        # Model configuration
        st.markdown("### ⚙️ Model Configuration")
        config_cols = st.columns(3)
        
        with config_cols[0]:
            st.write(f"**Image Size:** {config.image_size[0]}×{config.image_size[1]}")
            st.write(f"**Num Classes:** 10")
        
        with config_cols[1]:
            st.write(f"**Conv Filters:** (32, 64, 128)")
            st.write(f"**Dense Units:** 128")
        
        with config_cols[2]:
            dropout_rate = selected_model.split()[-1]  # Extract "0.1", "0.3", or "0.5"
            st.write(f"**Dropout Rate:** {dropout_rate}")
            st.write(f"**Source:** Weights & Biases (W&B)")

if __name__ == "__main__":
    main()
