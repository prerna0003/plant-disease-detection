import os
import pickle
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request

import tensorflow as tf

app = Flask(__name__)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "fine_tuned_plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.environ.get(
    "CLASS_NAMES_PATH",
    "class_names.pkl"
)

IMAGE_SIZE = (224, 224)

# ---------------------------------------------------------
# Load model and class names
# ---------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "rb") as f:
    class_names = pickle.load(f)

class_names = list(class_names)

# ---------------------------------------------------------
# Disease information
# ---------------------------------------------------------
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "status": "Disease",
        "description": "A bacterial disease that causes dark spots and lesions on pepper leaves and fruit.",
        "tips": "Remove infected plant material, avoid overhead watering, improve air circulation, and use clean planting material."
    },
    "Pepper__bell___healthy": {
        "plant": "Bell Pepper",
        "status": "Healthy",
        "description": "The uploaded bell pepper leaf is classified as healthy.",
        "tips": "Continue regular monitoring, proper watering, balanced nutrition, and good airflow."
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "status": "Disease",
        "description": "Early blight commonly produces dark circular or target-like lesions on potato leaves.",
        "tips": "Remove heavily infected foliage, improve airflow, avoid prolonged leaf wetness, and follow appropriate crop-management practices."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "status": "Disease",
        "description": "Late blight can produce irregular dark lesions and rapidly damage potato foliage.",
        "tips": "Remove affected material, reduce leaf wetness, maintain airflow, and use appropriate disease-management practices."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "status": "Healthy",
        "description": "The uploaded potato leaf is classified as healthy.",
        "tips": "Continue routine monitoring and maintain good irrigation, nutrition, and field hygiene."
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "status": "Disease",
        "description": "Bacterial spot can cause small dark lesions on tomato leaves and fruit.",
        "tips": "Avoid overhead irrigation, remove severely affected material, improve airflow, and use clean tools and planting material."
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "status": "Disease",
        "description": "Early blight is associated with dark lesions, often with concentric ring patterns, on tomato foliage.",
        "tips": "Remove affected leaves, improve airflow, avoid wetting foliage, and follow appropriate crop-management practices."
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "status": "Disease",
        "description": "Late blight can cause dark, water-soaked or irregular lesions on tomato leaves.",
        "tips": "Remove affected plant material, reduce leaf wetness, improve airflow, and use appropriate disease-management practices."
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "status": "Disease",
        "description": "Tomato leaf mold commonly develops under humid conditions and affects tomato foliage.",
        "tips": "Improve ventilation, reduce humidity around foliage, avoid overhead watering, and remove severely affected leaves."
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "status": "Disease",
        "description": "Septoria leaf spot causes numerous small circular spots on tomato leaves.",
        "tips": "Remove affected lower leaves, improve airflow, avoid splashing water onto foliage, and maintain garden hygiene."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "status": "Pest",
        "description": "Two-spotted spider mites can cause stippling, yellowing, and fine webbing on leaves.",
        "tips": "Inspect leaf undersides, reduce plant stress, wash foliage when appropriate, and use suitable integrated pest-management practices."
    },
    "Tomato__Target_Spot": {
        "plant": "Tomato",
        "status": "Disease",
        "description": "Target spot produces circular lesions that may develop concentric rings on tomato leaves.",
        "tips": "Improve airflow, avoid prolonged leaf wetness, remove severely affected foliage, and maintain plant hygiene."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "plant": "Tomato",
        "status": "Viral Disease",
        "description": "Tomato yellow leaf curl virus can cause leaf curling, yellowing, and reduced plant growth.",
        "tips": "Monitor for whiteflies, remove infected plants when appropriate, control vectors, and use resistant varieties where available."
    },
    "Tomato__Tomato_mosaic_virus": {
        "plant": "Tomato",
        "status": "Viral Disease",
        "description": "Tomato mosaic virus can cause mottled or mosaic patterns and abnormal leaf development.",
        "tips": "Remove infected material, disinfect tools, avoid handling plants with contaminated hands, and use clean planting material."
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "status": "Healthy",
        "description": "The uploaded tomato leaf is classified as healthy.",
        "tips": "Continue routine monitoring, appropriate watering, balanced nutrition, and good airflow."
    }
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def pretty_name(name):
    return (
        name.replace("___", " • ")
            .replace("__", " • ")
            .replace("_", " ")
    )


def predict_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # The trained model already contains its MobileNetV2
    # preprocessing layers, so no extra preprocess_input()
    # is applied here.
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index] * 100)

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


# ---------------------------------------------------------
# HTML template - single-file Flask UI
# ---------------------------------------------------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease AI</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: Inter, Arial, sans-serif;
            min-height: 100vh;
            background:
                radial-gradient(circle at top left, rgba(53, 130, 80, .25), transparent 32%),
                radial-gradient(circle at bottom right, rgba(31, 90, 58, .28), transparent 35%),
                #07130d;
            color: #eef8f1;
        }

        .container {
            width: min(1120px, 92%);
            margin: auto;
        }

        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 24px 0;
        }

        .brand {
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -.5px;
        }

        .brand span {
            color: #70d99a;
        }

        .badge {
            border: 1px solid rgba(255,255,255,.13);
            background: rgba(255,255,255,.05);
            padding: 8px 13px;
            border-radius: 999px;
            font-size: 12px;
            color: #b9d8c5;
        }

        .hero {
            text-align: center;
            padding: 55px 0 35px;
        }

        .hero .eyebrow {
            color: #70d99a;
            font-weight: 700;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 14px;
        }

        h1 {
            font-size: clamp(38px, 7vw, 70px);
            line-height: .98;
            letter-spacing: -3px;
            margin-bottom: 20px;
        }

        .hero p {
            max-width: 680px;
            margin: auto;
            color: #a9bcb0;
            font-size: 17px;
            line-height: 1.7;
        }

        .main-card {
            max-width: 850px;
            margin: 25px auto 70px;
            background: rgba(255,255,255,.065);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 28px;
            padding: 28px;
            box-shadow: 0 30px 80px rgba(0,0,0,.28);
            backdrop-filter: blur(18px);
        }

        .upload-box {
            border: 2px dashed rgba(112,217,154,.38);
            border-radius: 22px;
            min-height: 280px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 35px;
            text-align: center;
            background: rgba(112,217,154,.035);
        }

        .upload-icon {
            width: 72px;
            height: 72px;
            border-radius: 22px;
            display: grid;
            place-items: center;
            font-size: 31px;
            background: rgba(112,217,154,.12);
            margin-bottom: 18px;
        }

        .upload-box h2 {
            font-size: 21px;
            margin-bottom: 8px;
        }

        .upload-box p {
            color: #8fa398;
            margin-bottom: 22px;
            font-size: 14px;
        }

        input[type="file"] {
            display: none;
        }

        .upload-btn, .predict-btn {
            display: inline-block;
            cursor: pointer;
            border: 0;
            border-radius: 13px;
            padding: 13px 22px;
            font-size: 14px;
            font-weight: 800;
            transition: .2s ease;
        }

        .upload-btn {
            background: #70d99a;
            color: #07130d;
        }

        .upload-btn:hover, .predict-btn:hover {
            transform: translateY(-2px);
        }

        .selected-file {
            margin-top: 15px;
            color: #9bc7aa;
            font-size: 13px;
            min-height: 18px;
        }

        .predict-btn {
            width: 100%;
            margin-top: 20px;
            background: linear-gradient(135deg, #70d99a, #39b977);
            color: #06100a;
            font-size: 15px;
            padding: 15px;
        }

        .result {
            margin-top: 25px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
        }

        .preview {
            border-radius: 20px;
            overflow: hidden;
            background: #050c08;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .preview img {
            width: 100%;
            height: 100%;
            max-height: 430px;
            object-fit: contain;
        }

        .result-info {
            padding: 22px;
            border-radius: 20px;
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.08);
        }

        .result-label {
            color: #8fa398;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
        }

        .prediction {
            font-size: 27px;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 15px;
        }

        .status {
            display: inline-block;
            padding: 7px 11px;
            border-radius: 999px;
            background: rgba(112,217,154,.12);
            color: #70d99a;
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 20px;
        }

        .confidence-title {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: #a9bcb0;
            margin-bottom: 8px;
        }

        .progress {
            height: 10px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255,255,255,.09);
            margin-bottom: 23px;
        }

        .progress-bar {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #39b977, #8de8ad);
        }

        .info-block {
            margin-top: 18px;
        }

        .info-block h3 {
            font-size: 14px;
            margin-bottom: 7px;
        }

        .info-block p {
            color: #9fb1a6;
            line-height: 1.55;
            font-size: 13px;
        }

        .warning {
            margin-top: 15px;
            padding: 12px;
            border-radius: 12px;
            background: rgba(255, 185, 75, .09);
            color: #e8c98f;
            font-size: 12px;
            line-height: 1.5;
        }

        .footer {
            text-align: center;
            padding: 0 0 35px;
            color: #718078;
            font-size: 12px;
        }

        @media (max-width: 760px) {
            .hero {
                padding-top: 35px;
            }

            .main-card {
                padding: 16px;
                border-radius: 22px;
            }

            .result {
                grid-template-columns: 1fr;
            }

            h1 {
                letter-spacing: -2px;
            }
        }
    </style>
</head>

<body>
    <div class="container">

        <nav>
            <div class="brand">🌱 Plant<span>AI</span></div>
            <div class="badge">MobileNetV2 • 15 Classes</div>
        </nav>

        <section class="hero">
            <div class="eyebrow">AI-powered plant analysis</div>
            <h1>Detect plant diseases<br>in seconds.</h1>
            <p>
                Upload a clear image of a plant leaf and our deep learning
                model will identify the most likely disease or healthy class.
            </p>
        </section>

        <main class="main-card">

            <form method="POST" enctype="multipart/form-data">

                <div class="upload-box">
                    <div class="upload-icon">🍃</div>
                    <h2>Upload a leaf image</h2>
                    <p>PNG, JPG or JPEG • Recommended: clear leaf image</p>

                    <label for="image" class="upload-btn">
                        Choose Image
                    </label>

                    <input
                        id="image"
                        type="file"
                        name="image"
                        accept="image/png,image/jpeg,image/jpg"
                        required
                    >

                    <div id="filename" class="selected-file">
                        No image selected
                    </div>
                </div>

                <button class="predict-btn" type="submit">
                    🔍 Analyze Leaf
                </button>

            </form>

            {% if error %}
                <div class="warning">
                    ⚠️ {{ error }}
                </div>
            {% endif %}

            {% if result %}
            <div class="result">

                <div class="preview">
                    <img src="{{ image_data }}" alt="Uploaded plant leaf">
                </div>

                <div class="result-info">

                    <div class="result-label">Prediction</div>

                    <div class="prediction">
                        {{ pretty_prediction }}
                    </div>

                    <div class="status">
                        {{ disease_status }}
                    </div>

                    <div class="confidence-title">
                        <span>Model confidence</span>
                        <strong>{{ "%.2f"|format(confidence) }}%</strong>
                    </div>

                    <div class="progress">
                        <div
                            class="progress-bar"
                            style="width: {{ confidence }}%;"
                        ></div>
                    </div>

                    <div class="info-block">
                        <h3>About this result</h3>
                        <p>{{ description }}</p>
                    </div>

                    <div class="info-block">
                        <h3>Recommended next steps</h3>
                        <p>{{ tips }}</p>
                    </div>

                    {% if confidence < 70 %}
                    <div class="warning">
                        ⚠️ The model has relatively low confidence in this
                        prediction. Try a clearer image with the leaf centered
                        and well illuminated.
                    </div>
                    {% elif confidence < 85 %}
                    <div class="warning">
                        ℹ️ Moderate confidence. For a more reliable result,
                        try another clear image from a different angle.
                    </div>
                    {% endif %}

                </div>
            </div>
            {% endif %}

        </main>

        <div class="footer">
            PlantAI • Deep Learning Plant Disease Detection •
            MobileNetV2 • 94.91% validation accuracy
        </div>

    </div>

    <script>
        const imageInput = document.getElementById("image");
        const filename = document.getElementById("filename");

        imageInput.addEventListener("change", function () {
            if (this.files && this.files.length > 0) {
                filename.textContent = this.files[0].name;
            } else {
                filename.textContent = "No image selected";
            }
        });
    </script>
</body>
</html>
"""


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = False
    error = None
    image_data = None

    prediction = None
    pretty_prediction = None
    confidence = 0
    disease_status = None
    description = None
    tips = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "Please select an image."
            return render_template_string(
                HTML,
                result=False,
                error=error
            )

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image."
            return render_template_string(
                HTML,
                result=False,
                error=error
            )

        try:
            image = Image.open(file.stream).convert("RGB")

            prediction, confidence = predict_image(image)

            info = DISEASE_INFO.get(
                prediction,
                {
                    "plant": "Plant",
                    "status": "Classification",
                    "description": "The model generated a prediction for the uploaded image.",
                    "tips": "For best results, use a clear image of a single leaf."
                }
            )

            # Convert uploaded image to a browser-friendly data URL.
            import base64
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90)

            encoded_image = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            image_data = (
                "data:image/jpeg;base64,"
                + encoded_image
            )

            pretty_prediction = pretty_name(prediction)
            disease_status = info["status"]
            description = info["description"]
            tips = info["tips"]

            result = True

        except Exception as exc:
            error = (
                "Unable to process this image. "
                "Please upload a valid JPG or PNG leaf image."
            )
            print("Prediction error:", exc)

    return render_template_string(
        HTML,
        result=result,
        error=error,
        image_data=image_data,
        prediction=prediction,
        pretty_prediction=pretty_prediction,
        confidence=confidence,
        disease_status=disease_status,
        description=description,
        tips=tips
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
