from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

def create_presentation():
    # Create presentation
    prs = Presentation()
    
    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "AeroMinds: Working Implementation"
    subtitle.text = "AI-Powered Aerial Waste Intelligence\nDetect • Assess • Respond"
    
    # -------------------------------------------------------------
    # SLIDE 2: Project Mission
    # -------------------------------------------------------------
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    
    title_shape.text = "The Mission"
    tf = body_shape.text_frame
    tf.text = "AeroMinds transforms raw aerial drone footage into actionable environmental intelligence."
    
    p = tf.add_paragraph()
    p.text = "Leverages state-of-the-art YOLOv8 object detection."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Autonomously identifies illegal dumping sites."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Categorizes severity to streamline response pipelines."
    p.level = 1
    
    # -------------------------------------------------------------
    # SLIDE 3: Key Features
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    
    title_shape.text = "Key Features"
    tf = body_shape.text_frame
    
    features = [
        "Live Intelligence Dashboard: Command center for urban sanitation.",
        "Aerial Inference: Drag-and-drop for high-res images & videos.",
        "Severity Engine: Calculates waste coverage and cluster counts.",
        "Automated Pipeline: Tracks detections (DETECTED → CLEARED).",
        "Hardware Accelerated: PyTorch CUDA pipeline."
    ]
    
    tf.text = features[0]
    for feature in features[1:]:
        p = tf.add_paragraph()
        p.text = feature
        
    # -------------------------------------------------------------
    # SLIDE 4: Working Implementation Details
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    
    title_shape.text = "Working Implementation: The Dataset & Model"
    tf = body_shape.text_frame
    
    tf.text = "Model Architecture:"
    
    p = tf.add_paragraph()
    p.text = "Custom YOLOv8n fine-tuned model"
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Dataset: Roboflow Aerial-Dumping-Sites (v6)"
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Training Workflow established using train_yolo.py"
    p.level = 1

    p = tf.add_paragraph()
    p.text = "Model Deployment:"
    p.level = 0
    
    p = tf.add_paragraph()
    p.text = "Streamlit Interactive Dashboard"
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Flask fallback REST API server"
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Docker containerization for universal deployment"
    p.level = 1
    
    # -------------------------------------------------------------
    # SLIDE 5: Next Steps & Impact
    # -------------------------------------------------------------
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    
    title_shape.text = "Environmental Impact"
    tf = body_shape.text_frame
    
    tf.text = "Continuous aerial monitoring prevents wide-scale pollution."
    p = tf.add_paragraph()
    p.text = "Faster response times to illegal dumping activities."
    p = tf.add_paragraph()
    p.text = "Built to keep our environment clean, one flight at a time."
    
    # Save presentation
    output_filename = "AeroMinds_Working_Implementation.pptx"
    prs.save(output_filename)
    print(f"Presentation generated successfully: {output_filename}")

if __name__ == "__main__":
    create_presentation()
