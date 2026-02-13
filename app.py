import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageOps
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
import gc
import traceback
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cctv-vqa-system-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global model variables
blip_processor = None
blip_model = None

def load_blip_model():
    """Load BLIP model for VQA"""
    global blip_processor, blip_model
    try:
        print("Loading BLIP VQA model...")
        
        # Using smaller BLIP model for faster inference
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        
        blip_model.to(device)
        blip_model.eval()
        print("BLIP model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading BLIP model: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhance_cctv_image(image):
    """Enhance CCTV images for better recognition"""
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)
        
        # Enhance brightness if image is dark
        mean_brightness = sum(image.convert('L').getdata()) / len(image.convert('L').getdata())
        if mean_brightness < 100:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.4)
        
        return image
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return image

def get_blip_answer(image, question):
    """Get answer using BLIP model"""
    try:
        # Prepare inputs
        inputs = blip_processor(image, question, return_tensors="pt").to(device)
        
        # Generate answer
        with torch.no_grad():
            outputs = blip_model.generate(**inputs)
        
        # Decode answer
        answer = blip_processor.decode(outputs[0], skip_special_tokens=True)
        
        return answer.strip()
    except Exception as e:
        print(f"BLIP inference error: {e}")
        return None

def post_process_answer(answer, question):
    """Post-process answers for better accuracy"""
    if not answer:
        return "Unable to determine"
    
    answer_lower = answer.lower().strip()
    question_lower = question.lower().strip()
    
    # Common CCTV object mappings
    object_mappings = {
        # Phone variations
        'cell phone': 'phone', 'mobile phone': 'phone', 'cellphone': 'phone',
        'smartphone': 'phone', 'phone': 'phone', 'mobile': 'phone',
        
        # Weapon variations
        'knife': 'knife', 'dagger': 'knife', 'blade': 'knife', 'sword': 'knife',
        'gun': 'gun', 'pistol': 'gun', 'firearm': 'gun', 'rifle': 'gun',
        'bat': 'bat', 'baseball bat': 'bat', 'club': 'bat',
        
        # Bag variations
        'bag': 'bag', 'backpack': 'bag', 'purse': 'bag', 'handbag': 'bag',
        'suitcase': 'bag', 'briefcase': 'bag',
        
        # Tools
        'tool': 'tool', 'hammer': 'hammer', 'screwdriver': 'screwdriver',
        'wrench': 'wrench', 'pliers': 'pliers',
        
        # Electronics
        'tablet': 'tablet', 'laptop': 'laptop', 'camera': 'camera',
        'remote': 'remote control',
        
        # Personal items
        'wallet': 'wallet', 'keys': 'keys', 'umbrella': 'umbrella',
        'newspaper': 'newspaper', 'book': 'book',
        
        # Food/drink
        'bottle': 'bottle', 'cup': 'cup', 'glass': 'glass',
        'sandwich': 'sandwich', 'food': 'food',
        
        # Nothing/empty
        'nothing': 'nothing', 'empty': 'nothing', 'none': 'nothing',
        "don't see": 'nothing', 'no object': 'nothing',
    }
    
    # Specific handling for "holding" questions
    if 'holding' in question_lower or 'hold' in question_lower or 'carrying' in question_lower:
        holding_keywords = ['holding', 'has', 'carrying', 'with', 'in hand', 'holding a', 'holding an']
        
        # First, check if answer contains known objects
        for key, value in object_mappings.items():
            if key in answer_lower:
                return value
        
        # If no match found, try to extract object from sentence structure
        words = answer_lower.split()
        if len(words) > 2:
            # Look for patterns like "he is holding X" or "man has X"
            for i, word in enumerate(words):
                if word in ['holding', 'has', 'carrying'] and i + 1 < len(words):
                    next_word = words[i + 1]
                    for key, value in object_mappings.items():
                        if key.startswith(next_word):
                            return value
    
    # General object matching
    for key, value in object_mappings.items():
        if key in answer_lower:
            return value
    
    # If answer is very short or seems incomplete
    if len(answer_lower.split()) <= 2:
        # Check if it might be an object name
        common_objects = ['phone', 'knife', 'bag', 'gun', 'bat', 'wallet', 'keys']
        for obj in common_objects:
            if obj in answer_lower:
                return obj
    
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'url': f'/static/uploads/{unique_filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        filename = data.get('filename')
        question = data.get('question')
        
        if not filename or not question:
            return jsonify({'error': 'Missing filename or question'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        # Load and preprocess image
        image = Image.open(filepath).convert('RGB')
        enhanced_image = enhance_cctv_image(image)
        
        # Get answer from BLIP model
        if blip_model is None:
            load_blip_model()
        
        if blip_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Generate answer
        raw_answer = get_blip_answer(enhanced_image, question)
        
        if raw_answer:
            # Post-process answer
            final_answer = post_process_answer(raw_answer, question)
            
            return jsonify({
                'success': True,
                'answer': final_answer,
                'raw_answer': raw_answer,
                'question': question
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate answer'
            }), 500
            
    except Exception as e:
        print(f"Error in ask_question: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/examples', methods=['GET'])
def get_examples():
    examples = [
        {'question': 'What is the person holding?', 'description': 'Identify objects in hand'},
        {'question': 'How many people are in the image?', 'description': 'Count persons'},
        {'question': 'What color is the vehicle?', 'description': 'Identify vehicle color'},
        {'question': 'Is there any weapon visible?', 'description': 'Detect dangerous objects'},
        {'question': 'What is the person wearing?', 'description': 'Describe clothing'},
        {'question': 'Is the person wearing a mask?', 'description': 'Check face coverings'},
        {'question': 'What type of vehicle is this?', 'description': 'Classify vehicle'},
        {'question': 'What is the main activity?', 'description': 'Describe the scene'},
        {'question': 'Is it day or night?', 'description': 'Determine time period'},
        {'question': 'Are there any bags visible?', 'description': 'Detect luggage/bags'}
    ]
    return jsonify({'examples': examples})

@app.route('/health', methods=['GET'])
def health_check():
    """Check if model is loaded"""
    model_loaded = blip_model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model not loaded',
        'model_loaded': model_loaded,
        'device': device
    })

if __name__ == '__main__':
    # Load model on startup
    print("Starting CCTV VQA System...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    # Load model
    load_blip_model()
    
    # Start server
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)