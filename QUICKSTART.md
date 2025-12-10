# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Data

```bash
python src/data/synthetic_data_generator.py
```

This creates `data/raw/wind_turbine_scada.csv` with data for 10 turbines over 1 year.

### Step 3: Run the Experiment

```bash
python run_experiment.py
```

This will:
- Load and preprocess data
- Create features
- Train models
- Generate results

### Step 4: Launch Dashboard

```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## 📓 Explore Notebooks

```bash
jupyter notebook notebooks/
```

1. **01_exploration.ipynb** - Data exploration
2. **02_modeling_baselines.ipynb** - Baseline models (create from template)
3. **03_deep_learning_and_RUL.ipynb** - Deep learning (create from template)

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## ⚙️ Configuration

Edit `src/config/config.yaml` to customize:
- Model hyperparameters
- Maintenance thresholds
- Data paths

---

## 📊 Expected Output

After running `run_experiment.py`, you should see:
- `results/predictions.csv` - Model predictions
- `results/maintenance_recommendations.csv` - Maintenance schedule
- `models/saved/` - Trained models
- `logs/project.log` - Execution logs

---

## ❓ Troubleshooting

**Issue**: Import errors
**Solution**: Ensure you're in the project root directory

**Issue**: Data not found
**Solution**: Run `python src/data/synthetic_data_generator.py` first

**Issue**: CUDA errors
**Solution**: Code works on CPU. GPU is optional.

---

## 📚 Next Steps

1. Read `README.md` for detailed documentation
2. Explore `notebooks/` for detailed analysis
3. Review `reports/` for project structure
4. Customize `config.yaml` for your needs

---

**Happy Coding!** 🎉

