# DeepTrace: Forensic Graphs for AI-Generated Image Detection

DeepTrace is a Python-powered forensic intelligence system that detects AI-generated (deepfake) images using handcrafted visual features and graph-based anomaly detection.

## 🚀 Features
- Extracts forensic features (edges, noise residuals, blockiness, GLCM textures, FFT).
- Builds a similarity graph of images using cosine similarity.
- Uses DBSCAN clustering and Isolation Forest for anomaly detection.
- Generates visual outputs: scatter plots, feature histograms, similarity graphs.
- Produces a precision/recall report.

## 📊 Example Results
- Tested on 40 images (20 AI, 20 real).
- Achieved **65% precision, 65% recall** at balanced threshold.
- Flexible tuning for **high recall** (security triage) or **high precision** (low false alarms).

## 🛠️ Tech Stack
- Python 3.10+
- OpenCV, NumPy, Pandas, Matplotlib
- Scikit-learn, NetworkX, scikit-image

## ▶️ How to Run
```bash
# clone repo
git clone https://github.com/<your-username>/deeptrace.git
cd deeptrace

# create venv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# install requirements
pip install -r requirements.txt

# run with sample data
python deeptrace.py --data_dir ./data --out_dir ./outputs --size 256 --knn 8 --anom_thresh 0.65
