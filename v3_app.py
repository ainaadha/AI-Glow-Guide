import streamlit as st
import os
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from openai import OpenAI
import base64
from PIL import Image
import io
import ast
import json
from dotenv import load_dotenv
import time

load_dotenv()

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="GlowGuide - AI Skin Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 20px auto;
        max-width: 1200px;
    }
    
    h1 {
        text-align: center;
        color: #667eea;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 40px;
        font-size: 1.1em;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 40px;
        border: none;
        border-radius: 30px;
        font-size: 1.1em;
        font-weight: bold;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .detection-card {
        background: #f8f9ff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .product-card {
        background: white;
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }
    
    .product-name {
        color: #667eea;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .ingredients {
        background: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    
    .ai-analysis {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "D:\\06 - ACIE Project\\YOLOv11_Skin_Detection_Project\\Run_50_Epochs\\weights\\best.pt"
RECOMMENDATIONS_CSV = "cleaned.csv"
MAX_IMAGE_SIZE = (640, 640)

# ===============================
# INGREDIENT KNOWLEDGE BASE
# ===============================
INGREDIENT_BENEFITS = {
    "sodium chloride": ["oily", "acne"],
    "salicylic acid": ["acne", "oily", "blackheads"],
    "niacinamide": ["acne", "wrinkles", "redness", "hyperpigmentation"],
    "retinol": ["wrinkles", "acne", "aging", "fine lines"],
    "hyaluronic acid": ["dry", "wrinkles", "dehydration"],
    "sodium hyaluronate": ["dry", "wrinkles", "dehydration"],
    "aloe barbadenis": ["sensitive", "redness", "hydration", "irritation"],
    "aloe vera": ["sensitive", "redness", "hydration", "irritation"],
    "panthenol": ["dry", "sensitive", "hydration"],
    "glycolic acid": ["acne", "oily", "dullness", "texture"],
    "ceramide": ["dry", "sensitive", "barrier repair"],
    "vitamin c": ["dullness", "wrinkles", "hyperpigmentation", "brightening"],
    "ascorbic acid": ["dullness", "wrinkles", "hyperpigmentation", "brightening"],
    "benzoyl peroxide": ["acne", "bacterial"],
    "tea tree oil": ["acne", "oily", "bacterial"],
    "centella asiatica": ["sensitive", "redness", "irritation"],
    "peptides": ["wrinkles", "aging", "firmness"],
    "azelaic acid": ["acne", "redness", "hyperpigmentation"],
    "zinc": ["acne", "oily", "inflammation"],
    "collagen": ["wrinkles", "aging", "firmness"],
    "lactic acid": ["dry", "dullness", "texture"],
    "shea butter": ["dry", "sensitive", "hydration"],
    "squalane": ["dry", "hydration", "all skin types"],
    "glycerin": ["dry", "hydration", "sensitive"],
    "capric triglyceride": ["dry", "hydration"],
    "cetyl alcohol": ["dry", "emollient"],
    "stearyl alcohol": ["dry", "emollient"],
    "behentrimonium methosulfate": ["dry", "conditioning"],
}

# ===============================
# LOAD MODEL AND DATA
# ===============================
@st.cache_resource
def load_model():
    """Load YOLO model (cached)"""
    print("Loading YOLO model...")
    start_time = time.time()
    model = YOLO(MODEL_PATH)
    print(f"YOLO model loaded in {time.time() - start_time:.2f} seconds")
    return model

@st.cache_data
def load_skincare_data():
    """Load skincare dataset (cached)"""
    print("Loading skincare dataset...")
    start_time = time.time()
    df = pd.read_csv(RECOMMENDATIONS_CSV)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds ({len(df)} products)")
    return df

# Initialize
model = load_model()
skincare_df = load_skincare_data()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")

# ===============================
# HELPER FUNCTIONS
# ===============================
def get_ai_recommendations(skin_conditions, available_products):
    """Generate AI recommendations using OpenAI API"""
    prompt = f"""
You are a certified dermatologist and skincare expert. Based on the detected skin conditions and scientifically-matched products, provide personalized skincare recommendations.

Detected Skin Conditions:
{skin_conditions}

Available Skincare Products (matched by active ingredients):
{available_products}

Please provide your response in JSON format with this structure:
{{
  "analysis": "Brief analysis of the detected skin conditions",
  "top_recommendations": [
    {{
      "product_name": "Product name",
      "reason": "Scientific explanation of why this product is suitable, mentioning specific active ingredients"
    }}
  ]
}}

Focus on the top 3-5 most effective products. Be concise, friendly, and scientifically accurate.
"""
    
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a certified dermatologist and skincare expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return json.dumps({
            "analysis": "Unable to generate AI analysis at this time.",
            "top_recommendations": []
        })

def resize_image(img_array, max_size=MAX_IMAGE_SIZE):
    """Resize image if it's too large to speed up processing"""
    h, w = img_array.shape[:2]
    if h > max_size[0] or w > max_size[1]:
        scale = min(max_size[0]/h, max_size[1]/w)
        new_h, new_w = int(h*scale), int(w*scale)
        return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_array

def process_image(image):
    """Process image and detect skin conditions using YOLO"""
    try:
        start_time = time.time()
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize if too large
        img_array = resize_image(img_array)
        print(f"Image size: {img_array.shape}")
        
        # Run YOLO detection
        yolo_start = time.time()
        results = model(img_array, verbose=False)
        print(f"YOLO inference took {time.time() - yolo_start:.2f}s")
        
        # Extract detected conditions
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                detections.append({
                    'condition': class_name,
                    'confidence': round(confidence * 100, 2)
                })
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        print(f"Total image processing: {time.time() - start_time:.2f}s")
        return detections, annotated_img
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return [], None

def filter_products_by_ingredients(df, skin_conditions):
    """Filter products based on ingredient knowledge base"""
    start_time = time.time()
    recommended = []
    
    # Normalize conditions to lowercase
    conditions_lower = [cond.lower() for cond in skin_conditions]
    
    # Pre-filter relevant ingredients for detected conditions
    relevant_ingredients = {}
    for ing, benefits in INGREDIENT_BENEFITS.items():
        if any(cond in b.lower() or cond == b.lower() for b in benefits for cond in conditions_lower):
            relevant_ingredients[ing] = benefits
    
    print(f"Filtering products for conditions: {conditions_lower}")
    print(f"Relevant ingredients: {list(relevant_ingredients.keys())}")
    
    for idx, row in df.iterrows():
        try:
            # Parse ingredients from clean_ingreds column
            ingredients = []
            if 'clean_ingreds' in row and pd.notna(row['clean_ingreds']):
                try:
                    ingredients = [str(i).lower().strip() for i in ast.literal_eval(row['clean_ingreds'])]
                except Exception:
                    ingredients = [str(i).strip().lower() for i in str(row['clean_ingreds']).split(',')]
            
            if not ingredients:
                continue
            
            matched_ingredients = []
            relevance_score = 0
            
            for ing_key, benefits in relevant_ingredients.items():
                # Check if ingredient is present in product
                if any(ing_key in ing for ing in ingredients):
                    matched_ingredients.append(ing_key)
                    # Increase score for each benefit matching the detected conditions
                    for b in benefits:
                        for cond in conditions_lower:
                            if cond in str(b).lower():
                                relevance_score += 1
            
            if matched_ingredients:
                recommended.append({
                    "product_name": row.get("product_name", "Unknown"),
                    "brand_name": row.get("brand_name", "Unknown Brand"),
                    "matched_ingredients": matched_ingredients,
                    "relevance_score": relevance_score,
                    "price": row.get("price", "N/A"),
                    "product_url": row.get("product_url", ""),
                    "product_type": row.get("product_type", ""),
                })
                
                # Early exit if we have enough products
                if len(recommended) >= 30:
                    break
        
        except Exception as e:
            continue
    
    # Sort by relevance score (highest first)
    recommended = sorted(recommended, key=lambda x: x['relevance_score'], reverse=True)
    
    print(f"Product filtering took {time.time() - start_time:.2f}s, found {len(recommended)} products")
    
    return recommended[:10]  # Return top 10

def get_product_recommendations(detections):
    """Get product recommendations based on detected skin conditions"""
    # Extract unique conditions
    skin_conditions = list(set([d['condition'] for d in detections]))
    
    # Filter products by ingredients
    filtered_products = filter_products_by_ingredients(skincare_df, skin_conditions)
    
    return filtered_products

# ===============================
# STREAMLIT APP
# ===============================
# Title and subtitle
st.markdown("<h1>✨ AI Glow Guide</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Skin Analysis & Personalized Skincare Recommendations</p>", unsafe_allow_html=True)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'results' not in st.session_state:
    st.session_state.results = None

# File uploader
uploaded_file = st.file_uploader("📸 Upload Your Photo", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

# Display uploaded image
if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Analyze and Reset buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Analyze Skin", use_container_width=True):
            with st.spinner("Analyzing your skin... This may take a moment."):
                overall_start = time.time()
                
                # Process image
                print("\n=== Starting Analysis ===")
                detections, annotated_image = process_image(image)
                
                if not detections:
                    st.error("No skin conditions detected")
                    st.session_state.analyzed = False
                else:
                    # Get recommendations
                    filtered_products = get_product_recommendations(detections)
                    
                    # Format data for AI
                    conditions_text = "\n".join([
                        f"- {d['condition']} (Confidence: {d['confidence']}%)" 
                        for d in detections
                    ])
                    
                    products_text = "\n".join([
                        f"- {p['product_name']} ({p['brand_name']}) - Price: {p['price']}\n  Key Ingredients: {', '.join(p['matched_ingredients'])}"
                        for p in filtered_products[:5]
                    ]) if filtered_products else "No matching products found."
                    
                    # Generate AI recommendations
                    print("Calling OpenAI API...")
                    api_start = time.time()
                    ai_response = get_ai_recommendations(conditions_text, products_text)
                    print(f"OpenAI API took {time.time() - api_start:.2f}s")
                    
                    # Parse AI response
                    try:
                        ai_recommendations = json.loads(ai_response)
                    except:
                        ai_recommendations = {"raw_response": ai_response}
                    
                    print(f"=== Total analysis time: {time.time() - overall_start:.2f}s ===\n")
                    
                    # Store results
                    st.session_state.results = {
                        'detections': detections,
                        'annotated_image': annotated_image,
                        'product_details': filtered_products[:5] if filtered_products else [],
                        'ai_recommendations': ai_recommendations,
                        'processing_time': round(time.time() - overall_start, 2)
                    }
                    st.session_state.analyzed = True
                    st.rerun()
    
    with col3:
        if st.button("🔄 Upload New Image", use_container_width=True):
            st.session_state.analyzed = False
            st.session_state.results = None
            st.rerun()

# Display results
if st.session_state.analyzed and st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    
    # Detection results and annotated image
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Detection Results")
        if results['annotated_image'] is not None:
            st.image(results['annotated_image'], use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Detected Conditions")
        for detection in results['detections']:
            st.markdown(f"""
            <div class="detection-card">
                <strong style="color: #667eea; font-size: 1.1em;">{detection['condition']}</strong><br>
                Confidence: {detection['confidence']}%
            </div>
            """, unsafe_allow_html=True)
    
    # AI Analysis
    if results['ai_recommendations']:
        st.markdown("---")
        st.markdown("### 🤖 AI Dermatologist Analysis")
        
        ai_rec = results['ai_recommendations']
        
        if 'analysis' in ai_rec:
            st.markdown(f"""
            <div class="ai-analysis">
                <p>{ai_rec['analysis']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'top_recommendations' in ai_rec and ai_rec['top_recommendations']:
            st.markdown("#### 💡 Top Recommendations:")
            for idx, rec in enumerate(ai_rec['top_recommendations'], 1):
                st.markdown(f"""
                <div class="product-card">
                    <strong>{idx}. {rec.get('product_name', 'Product')}</strong>
                    <p style="margin-top: 5px; color: #666;">{rec.get('reason', '')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Product recommendations
    if results['product_details']:
        st.markdown("---")
        st.markdown("### 🛍️ Recommended Products")
        
        for idx, product in enumerate(results['product_details'], 1):
            ingredients_html = ""
            if product.get('matched_ingredients'):
                ingredients_html = f"""
                <div class="ingredients">
                    <strong>Key Ingredients:</strong> {', '.join(product['matched_ingredients'])}
                </div>
                """
            
            product_link = ""
            if product.get('product_url'):
                product_link = f'<a href="{product["product_url"]}" target="_blank" style="color: #667eea; font-weight: bold; text-decoration: none;">View Product →</a>'
            
            st.markdown(f"""
            <div class="product-card">
                <div class="product-name">{idx}. {product.get('product_name', 'Unknown Product')}</div>
                <div style="color: #666; margin: 5px 0;">
                    <strong>Brand:</strong> {product.get('brand_name', 'N/A')}
                </div>
                <div style="color: #666; margin: 5px 0;">
                    <strong>Type:</strong> {product.get('product_type', 'N/A')}
                </div>
                <div style="color: #666; margin: 5px 0;">
                    <strong>Price:</strong> {product.get('price', 'N/A')}
                </div>
                {ingredients_html}
                <div style="margin-top: 10px;">
                    {product_link}
                </div>
            </div>
            """, unsafe_allow_html=True)