import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Wheat Leaf Disease Detection",
    layout="wide"
)
st.markdown("""
<style>

.prediction-box{
    background-color:#d4edda;
    padding:20px;
    border-radius:15px;
    border-left:8px solid #28a745;
    color:#000000;
}

.info-box{
    background-color:#ffffff;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
    color:#000000;
}

.treatment-box{
    background-color:#fff8e1;
    padding:20px;
    border-radius:15px;
    border-left:8px solid #ff9800;
    color:#000000;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

model = tf.keras.models.load_model(
    "final_mobilenet.keras",
    compile=False
)

# Force model initialization


# =====================================================
# LOAD CLASS NAMES
# =====================================================

with open("class_names.json", "r") as f:
    class_names = json.load(f)
    

# =====================================================
# DISEASE INFORMATION + TREATMENT
# =====================================================

disease_info = {

    "aphid": {
        "info1": "Aphids are tiny sap-sucking insects commonly found on wheat leaves and stems.",
        "info2": "Heavy infestations weaken plants and reduce grain yield.",
        "info3": "They can also transmit plant viruses between crops.",

        "treat1": "Apply recommended insecticides when infestation exceeds threshold levels.",
        "treat2": "Encourage natural predators such as ladybugs and lacewings.",
        "treat3": "Regularly inspect crops to detect outbreaks early."
    },

    "black rust": {
        "info1": "Black rust is a fungal disease caused by Puccinia graminis.",
        "info2": "It produces dark brown to black pustules on stems and leaves.",
        "info3": "Severe infections can significantly reduce wheat yield.",

        "treat1": "Use rust-resistant wheat varieties whenever possible.",
        "treat2": "Apply fungicides during early disease development.",
        "treat3": "Monitor fields regularly for disease spread."
    },

    "blast": {
        "info1": "Wheat blast is a destructive fungal disease affecting spikes and leaves.",
        "info2": "It causes bleaching of spikelets and grain loss.",
        "info3": "The disease spreads rapidly under warm and humid conditions.",

        "treat1": "Use certified disease-free seeds.",
        "treat2": "Apply fungicides at recommended growth stages.",
        "treat3": "Avoid sowing during highly favorable weather conditions."
    },

    "brown rust": {
        "info1": "Brown rust produces orange-brown pustules on wheat leaves.",
        "info2": "The disease reduces photosynthesis and plant vigor.",
        "info3": "It spreads quickly through airborne spores.",

        "treat1": "Grow resistant wheat cultivars.",
        "treat2": "Apply fungicides when symptoms first appear.",
        "treat3": "Maintain field monitoring during humid periods."
    },

    "common root rot": {
        "info1": "Common root rot affects roots and lower stems of wheat plants.",
        "info2": "Infected plants show poor growth and yellowing.",
        "info3": "The disease survives in crop residues and soil.",

        "treat1": "Practice crop rotation with non-host crops.",
        "treat2": "Improve soil drainage and field sanitation.",
        "treat3": "Use healthy certified seeds."
    },

    "fusarium head blight": {
        "info1": "This disease infects wheat heads during flowering.",
        "info2": "It can contaminate grains with harmful mycotoxins.",
        "info3": "Yield and grain quality are severely affected.",

        "treat1": "Use resistant wheat varieties.",
        "treat2": "Apply fungicides during flowering stage.",
        "treat3": "Avoid excessive crop residue accumulation."
    },

    "healthy": {
        "info1": "No visible symptoms of disease were detected.",
        "info2": "The leaf appears healthy and physiologically normal.",
        "info3": "Proper crop management practices are being maintained.",

        "treat1": "Continue regular monitoring of the crop.",
        "treat2": "Maintain proper irrigation and fertilization.",
        "treat3": "Follow preventive disease management practices."
    },

    "leaf blight": {
        "info1": "Leaf blight causes elongated brown lesions on wheat leaves.",
        "info2": "Severe infection results in premature drying of leaves.",
        "info3": "The disease reduces photosynthetic activity.",

        "treat1": "Apply appropriate fungicides.",
        "treat2": "Remove infected crop residues after harvest.",
        "treat3": "Use disease-resistant wheat varieties."
    },

    "mildew": {
        "info1": "Powdery mildew appears as white powder-like fungal growth.",
        "info2": "It mainly affects leaves and young shoots.",
        "info3": "Severe infection can reduce grain production.",

        "treat1": "Apply sulfur-based fungicides.",
        "treat2": "Improve air circulation around plants.",
        "treat3": "Avoid excessive nitrogen fertilization."
    },

    "mite": {
        "info1": "Mites are microscopic pests that feed on plant tissues.",
        "info2": "Infested leaves often show yellowing and discoloration.",
        "info3": "Heavy infestations weaken plant growth.",

        "treat1": "Use recommended miticides when necessary.",
        "treat2": "Maintain field hygiene and weed control.",
        "treat3": "Monitor plants regularly for early infestation."
    },

    "septoria": {
        "info1": "Septoria leaf blotch causes irregular brown lesions on leaves.",
        "info2": "Lesions often contain tiny black fungal structures.",
        "info3": "The disease thrives under wet conditions.",

        "treat1": "Practice crop rotation.",
        "treat2": "Apply fungicides when disease pressure is high.",
        "treat3": "Use resistant cultivars where available."
    },

    "smut": {
        "info1": "Smut replaces healthy grains with black fungal spores.",
        "info2": "The disease affects grain quality and yield.",
        "info3": "Spores can survive on seeds and in soil.",

        "treat1": "Use fungicide-treated seeds.",
        "treat2": "Plant resistant wheat varieties.",
        "treat3": "Avoid using contaminated seed material."
    },

    "stem fly": {
        "info1": "Stem fly larvae feed inside wheat stems.",
        "info2": "Infested plants become weak and may lodge.",
        "info3": "The pest reduces overall crop productivity.",

        "treat1": "Use recommended insecticides.",
        "treat2": "Adopt timely sowing practices.",
        "treat3": "Remove heavily infested plants when possible."
    },

    "tan spot": {
        "info1": "Tan spot produces tan-colored lesions surrounded by yellow halos.",
        "info2": "The disease affects photosynthesis and plant health.",
        "info3": "It spreads through infected crop residues.",

        "treat1": "Apply fungicides when required.",
        "treat2": "Rotate crops to break disease cycles.",
        "treat3": "Remove infected residues after harvest."
    },

    "yellow rust": {
        "info1": "Yellow rust causes yellow stripe-like pustules on leaves.",
        "info2": "It spreads rapidly during cool and humid weather.",
        "info3": "Severe infection can cause substantial yield losses.",

        "treat1": "Apply fungicides immediately after detection.",
        "treat2": "Grow resistant wheat cultivars.",
        "treat3": "Inspect neighboring plants to prevent spread."
    }
}

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Wheat Disease Detection")

st.sidebar.write("""
### Models Used
- Deep CNN(MobileNetV2)
- Naive Bayes
- Decision Tree
- SVM
""")


# =====================================================
# MAIN TITLE
# =====================================================

st.title("Wheat Leaf Disease Detection System using Naive Bayes,Deep CNN, Decision Tree,SVM")

st.subheader(
    "Deep Learning and Statistical Texture Feature Analysis for Wheat Disease Classification"
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Wheat Leaf Image",
    type=["jpg", "png", "jpeg"]
)

# =====================================================
# GRAD CAM FUNCTION
# =====================================================
def generate_gradcam(img_array, model):

    # Last convolution layer directly from model
    last_conv_layer = model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = np.maximum(
        heatmap,
        0
    )

    max_val = np.max(heatmap)

    if max_val != 0:
        heatmap /= max_val

    return heatmap

# =====================================================
# PREDICTION SECTION
# =====================================================

if uploaded_file is not None:

    # ---------------- LOAD IMAGE ----------------

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # ---------------- SHOW ORIGINAL IMAGE ----------------

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )


    # ---------------- PREPROCESS ----------------

    resized_image = image.resize((160,160))

    img_array = np.array(resized_image)

    normalized_img = img_array / 255.0

    input_img = np.expand_dims(normalized_img, axis=0)

    # ---------------- PREDICTION ----------------

    predictions = model.predict(input_img)

    predicted_class = np.argmax(predictions)

    confidence = np.max(predictions)

    disease_name = class_names[predicted_class]

    lookup_name = disease_name.lower()

    # =====================================================
    import pandas as pd

    prob_df = pd.DataFrame({
    "Disease": class_names,
    "Probability": predictions[0]
    })

    prob_df = prob_df.sort_values(
    by="Probability",
    ascending=False
    )

    st.subheader("Prediction Probabilities")

    st.bar_chart(
    prob_df.set_index("Disease")
    )
    severity = confidence * 100

    st.metric(
    "Disease Confidence",
    f"{severity:.2f}%"
    )
    

# GRAD CAM
    # =====================================================

    heatmap = generate_gradcam(
        input_img,
        model
    )

    heatmap = cv2.resize(
        heatmap,
        (160,160)
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = heatmap * 0.4 + img_array

    # ---------------- SHOW HEATMAP ----------------

    with col2:

        st.subheader("Grad-CAM Heatmap")

        st.image(
            superimposed_img.astype("uint8"),
            use_container_width=True
        )

    # =====================================================
    # RESULTS
    # =====================================================

    st.markdown("---")

    st.markdown(f"""
    <div class="prediction-box">
    <h2>Predicted Disease</h2>
    <h1>{disease_name}</h1>
    <h3>Confidence: {confidence*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    
    st.subheader("Best Performing Model: Deep CNN (MobileNetV2)")

    # =====================================================
    # DISEASE INFO
    # =====================================================

    st.subheader("Disease Information")

    if lookup_name in disease_info:

        st.markdown(f"""
    <div class="info-box">

    <h3>🦠 Disease Information</h3>

    <ul>
    <li>{disease_info[lookup_name]["info1"]}</li>
    <li>{disease_info[lookup_name]["info2"]}</li>
    <li>{disease_info[lookup_name]["info3"]}</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

        # =====================================================
        # TREATMENT
        # =====================================================

        st.subheader("Treatment Recommendation")

        st.markdown(f"""
        <div class="treatment-box">

        <h3>💊 Treatment Recommendations</h3>

        <ul>
        <li>{disease_info[lookup_name]["treat1"]}</li>
        <li>{disease_info[lookup_name]["treat2"]}</li>
        <li>{disease_info[lookup_name]["treat3"]}</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)


        st.subheader("Farmer Recommendation")
        st.markdown("""
        <div class="info-box">

        <h3>👨‍🌾 Farmer Recommendation</h3>

        <ul>
        <li>Inspect nearby plants for similar symptoms.</li>
        <li>Monitor disease spread every 2–3 days.</li>
        <li>Follow integrated pest and disease management practices.</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.write(
            "Information currently unavailable."
        )
