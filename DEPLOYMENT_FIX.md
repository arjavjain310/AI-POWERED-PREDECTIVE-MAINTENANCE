# Fix for District Dropdown on Streamlit Cloud

## Issue
The 10 districts are not showing in the dropdown menu on the deployed Streamlit app.

## Root Cause
The data file (`data/raw/wind_turbine_scada_karnataka.csv`) is excluded from Git (via `.gitignore`), so it's not available on Streamlit Cloud. The app needs to generate data on first run.

## Solution Applied

### 1. Auto-Generation on First Run
The dashboard now automatically generates data with all 10 districts when the data file is not found:
- 30 turbines across 10 districts
- Full year of data (2023-01-01 to 2023-12-31)
- All districts: Chitradurga, Gadag, Davangere, Koppal, Vijayapura, Bagalkot, Belagavi, Raichur, Ballari, Tumakuru

### 2. Debug Information
Added debug messages in the sidebar to help identify issues:
- Shows if district column is missing
- Displays available columns
- Shows sample data if districts are not found

### 3. Multiple Path Checks
The app now checks multiple possible paths for the data file to work in different deployment environments.

## What Happens on Streamlit Cloud

1. **First Load (1-2 minutes):**
   - App detects data file is missing
   - Automatically generates data with all 10 districts
   - Shows "Generating Karnataka wind farm data..." message
   - Data is saved to `data/raw/wind_turbine_scada_karnataka.csv`

2. **Subsequent Loads:**
   - Uses the generated data file
   - All 10 districts appear in dropdown
   - Fast loading

## Verification

After deployment, check:
1. Sidebar shows "10 districts, 30 turbines" when "All Districts" is selected
2. Dropdown shows all 10 district names
3. Each district shows "3 turbines" when selected

## If Districts Still Don't Show

1. Check the sidebar for debug messages
2. Look for error messages in the app
3. Check Streamlit Cloud logs for errors
4. Verify data generation completed successfully

## Alternative: Commit Sample Data

If auto-generation is too slow, you can:
1. Generate a smaller sample file locally
2. Temporarily remove it from `.gitignore`
3. Commit and push the file
4. Restore `.gitignore`

```bash
# Generate smaller sample
python src/data/synthetic_data_generator.py

# Temporarily allow data file
# Edit .gitignore to comment out: data/raw/*.csv

# Commit and push
git add data/raw/wind_turbine_scada_karnataka.csv
git commit -m "Add sample data for deployment"
git push
```

