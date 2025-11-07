import gradio as gr
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
# CONFIGURATION
# ===============================
MODEL_PATH = "D:\\06 - ACIE Project\\YOLOv11_Skin_Detection_Project\\Run_50_Epochs\\weights\\best.pt" 
RECOMMENDATIONS_CSV = "cleaned.csv"  # Your skincare dataset
MAX_IMAGE_SIZE = (640, 640)

# ===============================
# GLOBAL VARIABLES (LOADED ONCE)
# ===============================
print("Loading YOLO model...")
start_time = time.time()
model = YOLO(MODEL_PATH)
print(f"YOLO model loaded in {time.time() - start_time:.2f} seconds")

print("Loading skincare dataset...")
start_time = time.time()
skincare_df = pd.read_csv(RECOMMENDATIONS_CSV)
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds ({len(skincare_df)} products)")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")

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


def analyze_skin(image):
    """Main function to analyze skin and provide recommendations"""
    try:
        if image is None:
            return None, "Please upload an image", ""
        
        overall_start = time.time()
        print("\n=== Starting Analysis ===")
        
        # Convert PIL Image to numpy array
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
        
        if not detections:
            # Get annotated image
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            return annotated_img, "No skin conditions detected in the image.", ""
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Get unique conditions
        skin_conditions = list(set([d['condition'] for d in detections]))
        
        # Filter products by ingredients
        filtered_products = filter_products_by_ingredients(skincare_df, skin_conditions)
        
        # Format data for AI
        conditions_text = "\n".join([
            f"- {d['condition']} (Confidence: {d['confidence']}%)" 
            for d in detections
        ])
        
        # Enhanced product text with matched ingredients
        products_text = "\n".join([
            f"- {p['product_name']} ({p['brand_name']}) - Price: {p['price']}\n  Key Ingredients: {', '.join(p['matched_ingredients'])}"
            for p in filtered_products[:5]
        ]) if filtered_products else "No matching products found."
        
        # Generate AI recommendations
        print("Calling OpenAI API...")
        api_start = time.time()
        ai_response = get_ai_recommendations(conditions_text, products_text)
        print(f"OpenAI API took {time.time() - api_start:.2f}s")
        
        # Try to parse JSON response
        try:
            ai_recommendations = json.loads(ai_response)
            analysis = ai_recommendations.get("analysis", "")
            recommendations = ai_recommendations.get("top_recommendations", [])
        except:
            analysis = ai_response
            recommendations = []
        
        # Format detection results
        detection_text = "**Detected Skin Conditions:**\n\n"
        for d in detections:
            detection_text += f"- **{d['condition']}** (Confidence: {d['confidence']}%)\n"
        
        detection_text += f"\n**Analysis:**\n{analysis}\n\n"
        
        # Format recommendations
        recommendations_text = "**Top Product Recommendations:**\n\n"
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                recommendations_text += f"**{i}. {rec.get('product_name', 'N/A')}**\n"
                recommendations_text += f"   {rec.get('reason', '')}\n\n"
        
        # Add detailed product information
        if filtered_products:
            recommendations_text += "\n**Product Details:**\n\n"
            for i, p in enumerate(filtered_products[:5], 1):
                recommendations_text += f"**{i}. {p['product_name']}** by {p['brand_name']}\n"
                recommendations_text += f"   - Price: {p['price']}\n"
                recommendations_text += f"   - Key Ingredients: {', '.join(p['matched_ingredients'])}\n"
                if p['product_url']:
                    recommendations_text += f"   - [View Product]({p['product_url']})\n"
                recommendations_text += "\n"
        
        print(f"=== Total analysis time: {time.time() - overall_start:.2f}s ===\n")
        
        return annotated_img, detection_text, recommendations_text
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", ""


# ===============================
# GRADIO INTERFACE
# ===============================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.output-image {
    border-radius: 10px;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="AI Skin Condition Analyzer") as demo:
    gr.Markdown(
        """
        # 🔬 AI Skin Condition Analyzer
        
        Upload a facial image to detect skin conditions and receive personalized skincare product recommendations.
        
        **How it works:**
        1. Upload a clear facial image
        2. AI analyzes and detects skin conditions
        3. Get personalized product recommendations based on active ingredients
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Facial Image",
                type="pil",
                height=400
            )
            analyze_btn = gr.Button("🔍 Analyze Skin", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### Tips for best results:
                - Use a clear, well-lit photo
                - Face should be clearly visible
                - Avoid heavy makeup for accurate detection
                """
            )
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Detected Conditions",
                type="numpy",
                height=400
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            detection_output = gr.Markdown(
                label="Skin Analysis",
                value="Upload an image and click 'Analyze Skin' to get started."
            )
        
        with gr.Column(scale=1):
            recommendations_output = gr.Markdown(
                label="Product Recommendations",
                value=""
            )
    
    # Connect the button to the function
    analyze_btn.click(
        fn=analyze_skin,
        inputs=[input_image],
        outputs=[output_image, detection_output, recommendations_output]
    )
    
    gr.Markdown(
        """
        ---
        **Note:** This tool is for educational purposes only and should not replace professional dermatological advice.
        """
    )

# Launch the app
if __name__ == "__main__":
    print("\n=== Starting Gradio Application ===")
    print(f"Model loaded: {model is not None}")
    print(f"Products loaded: {len(skincare_df)}")
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )