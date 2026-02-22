const video = document.getElementById('webcam');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

// Teeth DOM Elements
const predictedClassTeeth = document.getElementById('predicted-class-teeth');
const confidenceBarTeeth = document.getElementById('confidence-bar-teeth');
const confidenceTextTeeth = document.getElementById('confidence-text-teeth');

// Eye DOM Elements
const predictedClassEye = document.getElementById('predicted-class-eye');
const confidenceBarEye = document.getElementById('confidence-bar-eye');
const confidenceTextEye = document.getElementById('confidence-text-eye');

let tfliteModelTeeth;
let tfliteModelEye;
let classLabelsTeeth = [];
let classLabelsEye = [];
let isPredicting = false;

// Configurable constants
const MODEL_INPUT_SIZE = 224;
const PREDICTION_THRESHOLD = 0.50;

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            },
            audio: false
        });
        video.srcObject = stream;

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } catch (e) {
        alert("Camera access is required. Please allow camera permissions.");
        console.error("Camera access error:", e);
    }
}

async function loadLabels() {
    try {
        const responseTeeth = await fetch('./labels.txt');
        if (responseTeeth.ok) {
            const textAreaTeeth = await responseTeeth.text();
            classLabelsTeeth = textAreaTeeth.split('\n')
                .map(label => label.replace(/^\d+\s+/, '').trim())
                .filter(l => l.length > 0);
            console.log("Loaded Teeth Labels:", classLabelsTeeth);
        }

        const responseEye = await fetch('./label.txt');
        if (responseEye.ok) {
            const textAreaEye = await responseEye.text();
            classLabelsEye = textAreaEye.split('\n')
                .map(label => label.replace(/^\d+\s+/, '').trim())
                .filter(l => l.length > 0);
            console.log("Loaded Eye Labels:", classLabelsEye);
        }
    } catch (e) {
        console.warn("Could not load one or more label files. Using fallback placeholder labels.", e);
    }

    // Fallbacks just in case
    if (classLabelsTeeth.length === 0) classLabelsTeeth = ["Healthy", "Cavity", "Gingivitis", "Periodontitis", "Calculus"];
    if (classLabelsEye.length === 0) classLabelsEye = ["Cataract", "diabetic_retinopathy", "glaucoma", "Normal"];
}

async function loadModels() {
    loadingText.innerText = 'Loading TFLite Models...';
    tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');

    try {
        // Load Teeth model
        tfliteModelTeeth = await tflite.loadTFLiteModel('./model_unquant.tflite');
        console.log("Teeth Model loaded successfully", tfliteModelTeeth);

        // Load Eye model
        tfliteModelEye = await tflite.loadTFLiteModel('./Eye.tflite');
        console.log("Eye Model loaded successfully", tfliteModelEye);

        // Hide overlay once loaded
        loadingOverlay.classList.remove('active');

        // Start continuous inference
        predictLoop();
    } catch (e) {
        console.error("Error loading the models:", e);
        loadingText.innerText = 'Error: model_unquant.tflite or Eye.tflite not found.';
        document.querySelector('.spinner').style.display = 'none';
    }
}

async function predictLoop() {
    if (isPredicting) return;
    isPredicting = true;

    if (tfliteModelTeeth && tfliteModelEye) {
        // Run predictions for both models
        const modelOutputs = tf.tidy(() => {
            const imgTensor = tf.browser.fromPixels(video);
            const resized = tf.image.resizeBilinear(imgTensor, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
            const normalized = resized.div(255.0);
            const batched = normalized.expandDims(0);

            // Inference teeth
            const teethPredictionTensor = tfliteModelTeeth.predict(batched);
            // Inference eye
            const eyePredictionTensor = tfliteModelEye.predict(batched);

            return [teethPredictionTensor, eyePredictionTensor];
        });

        // Resolve data synchronously
        const outputTeethData = await modelOutputs[0].data();
        const outputEyeData = await modelOutputs[1].data();

        // Find max for teeth
        let highestProbTeeth = 0;
        let predictedLabelIndexTeeth = 0;
        for (let i = 0; i < outputTeethData.length; i++) {
            if (outputTeethData[i] > highestProbTeeth) {
                highestProbTeeth = outputTeethData[i];
                predictedLabelIndexTeeth = i;
            }
        }

        // Find max for eye
        let highestProbEye = 0;
        let predictedLabelIndexEye = 0;
        for (let i = 0; i < outputEyeData.length; i++) {
            if (outputEyeData[i] > highestProbEye) {
                highestProbEye = outputEyeData[i];
                predictedLabelIndexEye = i;
            }
        }

        // Dispose tensors
        modelOutputs[0].dispose();
        modelOutputs[1].dispose();

        // Update UI logic
        const predictedClassNameTeeth = classLabelsTeeth[predictedLabelIndexTeeth] || `Class ${predictedLabelIndexTeeth}`;
        const predictedClassNameEye = classLabelsEye[predictedLabelIndexEye] || `Class ${predictedLabelIndexEye}`;

        updateUI(predictedClassNameTeeth, highestProbTeeth, 'teeth');
        updateUI(predictedClassNameEye, highestProbEye, 'eye');
    }

    isPredicting = false;
    requestAnimationFrame(predictLoop);
}

function updateUI(className, confidence, type) {
    const confidencePct = Math.round(confidence * 100);

    // Choose which DOM elements to update based on the model type
    const classEl = type === 'teeth' ? predictedClassTeeth : predictedClassEye;
    const barEl = type === 'teeth' ? confidenceBarTeeth : confidenceBarEye;
    const textEl = type === 'teeth' ? confidenceTextTeeth : confidenceTextEye;

    if (confidence > PREDICTION_THRESHOLD) {
        classEl.innerText = className;
        barEl.style.width = `${confidencePct}%`;
        textEl.innerText = `${confidencePct}% Confidence`;

        const lowerClass = className.toLowerCase();

        // Define color logic based on keywords
        if (lowerClass.includes('healthy') || lowerClass.includes('normal')) {
            barEl.style.backgroundColor = 'var(--success-color)';
            classEl.style.color = 'var(--success-color)';
        } else if (lowerClass.includes('cavity') || lowerClass.includes('decay') || lowerClass.includes('caries') || lowerClass.includes('cataract') || lowerClass.includes('glaucoma') || lowerClass.includes('retinopathy')) {
            barEl.style.backgroundColor = 'var(--danger-color)';
            classEl.style.color = 'var(--danger-color)';
        } else {
            barEl.style.backgroundColor = 'var(--warning-color)';
            classEl.style.color = 'var(--warning-color)';
        }
    } else {
        classEl.innerText = 'Scanning...';
        classEl.style.color = 'var(--text-primary)';
        barEl.style.width = '0%';
        textEl.innerText = 'Waiting for clearer view...';
    }
}

// App Initialization
async function init() {
    await setupCamera();
    await loadLabels();
    await loadModels();
}

window.onload = init;
