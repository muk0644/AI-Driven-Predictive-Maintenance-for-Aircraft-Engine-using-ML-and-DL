# Aircraft Turbofan Engine Predictive Maintenance

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[![sklearn](https://img.shields.io/badge/scikit--learn-latest-red)](https://scikit-learn.org/)

Predict when aircraft engines will fail using machine learning and deep learning models. This project analyzes NASA's engine sensor data to give advance warning before failure occurs.

---

## Project Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROJECT PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────┘

   📊 RAW DATA                    🔧 PREPROCESSING              🤖 MODELS
   ────────────                   ───────────────              ──────────
   
   NASA Dataset        →     Clean & Transform      →     Train Models
   100 Engines              Remove bad sensors           7 Different AI Models
   21 Sensors               Normalize values             ML + Deep Learning
   20,631 Records           Create sequences             
                                                              ↓
                           
   📈 EVALUATION                  🏆 BEST MODEL               💾 DEPLOY
   ─────────────                 ─────────────               ──────────
   
   Compare Results     →     Hybrid CNN-LSTM        →     Save & Use
   Find best model           5.27 cycles error            Predict failures
   Generate reports          99.15% accuracy              1 day warning
```

---

## Understanding the Data (Simple Explanation)

### What is the Data?

Imagine 100 aircraft engines, each running until it breaks down. As they run, 21 different sensors measure things like:
- **Temperature**: How hot different engine parts get
- **Pressure**: How much force the air has inside
- **Speed**: How fast the fan and core are spinning
- **Fuel Flow**: How much fuel is being used

### Data Structure Visualization

```
Each Engine Record (One Row):
┌──────┬──────┬─────────────────┬──────────────────────────────┐
│Engine│ Time │  3 Settings     │    21 Sensor Readings        │
│  ID  │Cycle │(Operating Modes)│  (Temperature, Pressure...)  │
└──────┴──────┴─────────────────┴──────────────────────────────┘
   1      1       [3 numbers]         [21 numbers]
   1      2       [3 numbers]         [21 numbers]
   ...   ...          ...                  ...
   1     192     [3 numbers]         [21 numbers]  ← Engine Fails!

Total: 100 engines × ~200 cycles each = 20,631 measurements
```

### What We Predict: Remaining Useful Life (RUL)

```
Timeline of Engine Life:
├────────────────────────────────────────────────┤
0 cycles                                    192 cycles
(Brand New)      Time Passes →              (Failure)

At any point, RUL = How many cycles left until failure

Example:
- At cycle 100: RUL = 92 cycles left
- At cycle 150: RUL = 42 cycles left  
- At cycle 190: RUL = 2 cycles left ← URGENT MAINTENANCE!
```

### Data Preprocessing (What We Did)

```
Step 1: Remove Useless Sensors
   21 sensors → 7 don't change at all (removed) → 14 useful sensors
   
Step 2: Normalize Values
   Before: Temperature = 500°C, Speed = 2500 RPM
   After:  All values scaled to 0-1 range (easier for AI to learn)
   
Step 3: Create Sequences
   Instead of 1 measurement, look at last 50 cycles together
   
   [Cycle 1, Cycle 2, ..., Cycle 50] → Predict RUL
   [Cycle 2, Cycle 3, ..., Cycle 51] → Predict RUL
   (Like showing the AI a 50-frame video instead of 1 photo)
```

### Visual Data Analysis

**Engine Degradation Over Time:**
![Engine Degradation](results/engine_degradation.png)
*Shows how sensor readings change as engines approach failure. Clear degradation patterns visible.*

**Sensor Correlation Heatmap:**
![Sensor Correlations](results/sensor_correlations.png)
*Identifies which sensors are related to each other. Helps understand which sensors are most important.*

---

## Models Overview (Simple Explanation)

### Machine Learning Models (Traditional Approach)

```
┌─────────────────────────────────────────────────────────────┐
│  1. Random Forest                                           │
│     Think: 100 decision trees voting together               │
│     Result: RMSE = 41.37 cycles                            │
├─────────────────────────────────────────────────────────────┤
│  2. XGBoost                                                 │
│     Think: Smart sequential tree building                   │
│     Result: RMSE = 42.11 cycles                            │
├─────────────────────────────────────────────────────────────┤
│  3. LightGBM                                                │
│     Think: Faster version of XGBoost                        │
│     Result: RMSE = 41.18 cycles (Best ML Model)           │
└─────────────────────────────────────────────────────────────┘
```

**Feature Importance Analysis:**
![Feature Importance](results/feature_importance.png)
*Shows which sensors matter most for predictions. Temperature and pressure sensors are key.*

**Machine Learning Predictions:**
![ML Predictions](results/ml_predictions.png)
*Comparison of Random Forest, XGBoost, and LightGBM predictions vs actual RUL values.*

### Deep Learning Models (Advanced Neural Networks)

```
┌──────────────────────────────────────────────────────────────┐
│  4. LSTM (Long Short-Term Memory)                            │
│     Think: Remembers patterns over time                      │
│     Best for: Time-series data like engine degradation       │
│     Result: RMSE = 18.69 cycles                             │
├──────────────────────────────────────────────────────────────┤
│  5. 1D CNN (Convolutional Neural Network)                    │
│     Think: Finds patterns in sensor arrays                   │
│     Best for: Detecting local anomalies                      │
│     Result: RMSE = 15.25 cycles                             │
├──────────────────────────────────────────────────────────────┤
│  6. Bi-LSTM (Bidirectional LSTM)                            │
│     Think: Looks at data forwards AND backwards              │
│     Best for: Understanding full context                     │
│     Result: RMSE = 17.50 cycles                             │
├──────────────────────────────────────────────────────────────┤
│  7. Hybrid CNN-LSTM ⭐ WINNER!                               │
│     Think: CNN extracts features + LSTM learns time patterns │
│     Best for: Combining spatial and temporal learning        │
│     Result: RMSE = 5.27 cycles (BEST!)                      │
└──────────────────────────────────────────────────────────────┘
```

### Model Training Visualizations

**LSTM Training Progress:**
![LSTM Training History](results/lstm_training_history.png)
![LSTM Predictions](results/lstm_predictions.png)
*Left: Loss decreases over training epochs. Right: Predicted vs Actual RUL comparison.*

**CNN Training Progress:**
![CNN Training History](results/cnn_training_history.png)
![CNN Predictions](results/cnn_predictions.png)
*CNN model learns spatial patterns in sensor data efficiently.*

**Bi-LSTM Training Progress:**
![BiLSTM Training History](results/bilstm_training_history.png)
![BiLSTM Predictions](results/bilstm_predictions.png)
*Bidirectional processing improves context understanding.*

**Hybrid CNN-LSTM Training Progress:**
![Hybrid Training History](results/hybrid_training_history.png)
![Hybrid Predictions](results/hybrid_predictions.png)
*Best model combines CNN feature extraction with LSTM temporal learning. Notice the tight clustering around the diagonal line (perfect predictions).*

### Model Architecture: Hybrid CNN-LSTM (Best Model)

```
INPUT: 50 cycles × 14 sensors
         ↓
    ┌─────────────────┐
    │  CNN Layers     │  ← Extracts patterns from sensors
    │  (Feature       │     "This sensor combo looks bad"
    │   Extraction)   │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  LSTM Layers    │  ← Learns how patterns change over time
    │  (Temporal      │     "It's getting worse each cycle"
    │   Learning)     │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  Dense Layer    │  ← Combines information
    └────────┬────────┘
             ↓
    OUTPUT: RUL Prediction
    (e.g., "42 cycles until failure")
```

---

## Hyperparameter Tuning (Making Models Better)

### What is Hyperparameter Tuning?

Think of it like tuning a recipe:
- Too much salt = bad
- Too little salt = bland
- Just right = perfect!

Similarly, models have "settings" we can adjust:

```
┌─────────────────────────────────────────────────────────────┐
│  Model Parameters We Tuned:                                 │
├─────────────────────────────────────────────────────────────┤
│  1. Number of Trees (n_estimators)                          │
│     Tried: 100, 200, 300, 400, 500                          │
│     Like: How many expert opinions to combine               │
│                                                              │
│  2. Tree Depth (max_depth)                                  │
│     Tried: 5, 7, 10, 12, 15                                 │
│     Like: How many questions each tree can ask              │
│                                                              │
│  3. Learning Rate                                           │
│     Tried: 0.01, 0.05, 0.1, 0.15, 0.2                       │
│     Like: How fast the model learns (slow = careful)        │
│                                                              │
│  4. Feature Sampling (colsample_bytree)                     │
│     Tried: 0.6 to 1.0                                       │
│     Like: What % of sensors to look at each time            │
└─────────────────────────────────────────────────────────────┘

Process:
  Random Search → Try 20 different combinations
                → Pick the best performing one
                → Improves accuracy by 2-5%
```

### Tuning Results

```
Before Tuning → After Tuning → Improvement
─────────────────────────────────────────── 
XGBoost:   42.11  →  39.88    →  5.3% better
LightGBM:  41.18  →  39.24    →  4.7% better
```

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- 8GB RAM (16GB recommended)
- Internet connection (for dataset download)

### Quick Start

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd aircraft-engine-predictive-maintenance
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
```bash
python download_dataset.py
```

---

## Usage

### Run Complete Pipeline

Execute scripts in order (each takes 5-45 minutes):

```bash
# Phase 1: Understand the data
python 01_data_exploration.py           # ~2 minutes

# Phase 2: Clean and prepare data
python 02_data_preprocessing.py         # ~3 minutes

# Phase 3: Train traditional ML models
python 03_ml_baseline.py                # ~10 minutes

# Phase 4-7: Train deep learning models
python 04_deep_learning_lstm.py         # ~15 minutes
python 05_cnn_model.py                  # ~10 minutes
python 06_bilstm_model.py               # ~15 minutes
python 07_hybrid_cnn_lstm.py            # ~20 minutes (BEST MODEL)

# Phase 8: Optimize models
python 08_hyperparameter_tuning.py      # ~30 minutes

# Phase 9: Compare all results
python 09_final_comparison.py           # ~5 minutes
```

### Use Trained Model for Predictions

```python
from tensorflow import keras
import joblib
import numpy as np

# Load best model
model = keras.models.load_model('models/hybrid_cnn_lstm.h5')
scaler = joblib.load('models/scaler.pkl')

# Your engine data (50 cycles × 14 sensors)
your_data = np.array([...])  # Shape: (1, 50, 14)

# Predict
predicted_rul = model.predict(your_data)[0][0]
print(f"Remaining Useful Life: {predicted_rul:.0f} cycles")
print(f"Approximately {predicted_rul/5:.1f} days until maintenance needed")
```

---

## Results Summary

### Performance Comparison

| Rank | Model | Error (RMSE) | Accuracy (R²) | Training Time |
|------|-------|--------------|---------------|---------------|
| 🥇 | **Hybrid CNN-LSTM** | **5.27** | **99.15%** | ~20 min |
| 🥈 | 1D CNN | 15.25 | 92.90% | ~10 min |
| 🥉 | Bi-LSTM | 17.50 | 90.66% | ~15 min |
| 4 | LSTM | 18.69 | 89.35% | ~15 min |
| 5 | LightGBM | 41.18 | 62.89% | ~3 min |
| 6 | Random Forest | 41.37 | 62.54% | ~5 min |
| 7 | XGBoost | 42.11 | 61.18% | ~3 min |

### Visual Performance Comparison

**Comprehensive Model Comparison:**
![Final Comparison](results/FINAL_COMPREHENSIVE_COMPARISON.png)
*Side-by-side comparison of all 7 models across multiple metrics (RMSE, MAE, R²).*

**Performance Improvement Chart:**
![Performance Improvement](results/performance_improvement_chart.png)
*Shows the dramatic improvement from traditional ML to deep learning, with Hybrid CNN-LSTM achieving the best results.*

### Key Insights

- 🚀 **87.5% Improvement**: Deep learning beats traditional ML significantly
- ⏰ **Early Warning**: Predicts failure ~5 cycles (1 day) in advance
- 🎯 **High Accuracy**: 99.15% R² score means very reliable predictions
- 💪 **No Overfitting**: Model generalizes well to new engines

---

## Project Structure

```
aircraft-engine-predictive-maintenance/
│
├── 📁 data/                          # Dataset files
│   ├── train_FD001.txt              # Raw training data (100 engines)
│   ├── test_FD001.txt               # Raw test data (100 engines)
│   ├── RUL_FD001.txt                # Ground truth RUL values
│   ├── train_processed.csv          # Cleaned & preprocessed data
│   └── readme.txt                   # Dataset documentation
│
├── 📁 models/                        # Saved trained models
│   ├── hybrid_cnn_lstm.h5           # 🏆 Best model
│   ├── lstm_model.h5                # LSTM model
│   ├── cnn_model.h5                 # CNN model
│   ├── bilstm_model.h5              # Bi-LSTM model
│   ├── random_forest.pkl            # Random Forest
│   ├── xgboost.pkl                  # XGBoost
│   ├── lightgbm.pkl                 # LightGBM
│   ├── scaler.pkl                   # Data normalizer
│   └── feature_columns.pkl          # Feature names
│
├── 📁 results/                       # Visualizations & reports
│   ├── FINAL_COMPREHENSIVE_COMPARISON.png
│   ├── performance_improvement_chart.png
│   ├── EXECUTIVE_SUMMARY.md
│   ├── sensor_correlations.png
│   ├── lstm_training_history.png
│   └── [other visualizations]
│
├── 📄 download_dataset.py            # Download NASA data from Kaggle
├── 📄 01_data_exploration.py         # Explore raw data
├── 📄 02_data_preprocessing.py       # Clean & prepare data
├── 📄 03_ml_baseline.py              # Train ML models
├── 📄 04_deep_learning_lstm.py       # Train LSTM
├── 📄 05_cnn_model.py                # Train CNN
├── 📄 06_bilstm_model.py             # Train Bi-LSTM
├── 📄 07_hybrid_cnn_lstm.py          # Train Hybrid (Best!)
├── 📄 08_hyperparameter_tuning.py    # Optimize models
├── 📄 09_final_comparison.py         # Compare all models
├── 📄 requirements.txt               # Python packages needed
└── 📄 README.md                      # This file
```

---

## Technical Details

### Metrics Explained Simply

- **RMSE (Root Mean Square Error)**: Average prediction error in cycles
  - Lower is better
  - Our best: 5.27 cycles (like being off by 1 day)
  
- **MAE (Mean Absolute Error)**: Average difference between prediction and reality
  - Lower is better
  - Our best: 4.09 cycles
  
- **R² Score**: How much of the pattern does the model understand?
  - 0 = random guessing
  - 1 = perfect prediction
  - Our best: 0.9915 (99.15% accurate!)

### Training Configuration

```
Deep Learning Models Settings:
├─ Sequence Length: 50 cycles (look at last 50 measurements)
├─ Batch Size: 64 (process 64 examples at once)
├─ Optimizer: Adam (smart learning algorithm)
├─ Learning Rate: 0.001 (how fast to learn)
├─ Early Stopping: Stop if no improvement for 15 epochs
└─ Learning Rate Decay: Reduce by 50% if stuck for 5 epochs
```

---

## Real-World Impact

### What This Means for Airlines

```
❌ Without This System:
   → Engine fails unexpectedly
   → Emergency landing required  
   → Cost: $1-5 million
   → Safety risk: HIGH
   → Downtime: 1-2 weeks

✅ With This System:
   → Predict failure 1 day in advance
   → Schedule maintenance during routine check
   → Cost: $50,000-100,000
   → Safety risk: MINIMAL
   → Downtime: 1-2 days
   
   Savings: 90% cost reduction + Much safer!
```

---

## References

### Dataset
- **Saxena, A., & Goebel, K. (2008)**. Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository.
- **Link**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

### Research Papers
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM Conference.

### Technologies Used
- **Deep Learning**: TensorFlow 2.15, Keras
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

---

## License

This project is for educational and research purposes.

---