from nicegui import ui
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Path to the models
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.json"
WEIGHTS_PATH = BASE_DIR / "models" / "threshold_weights.npy"

try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    # Load the custom weights we found in Colab!
    threshold_weights = np.load(WEIGHTS_PATH)
except Exception as e:
    print(f"Error loading model or weights: {e}")

def predict_severity():
    input_data = pd.DataFrame([{
        'Temperature(F)': temp.value,
        'Humidity(%)': humidity.value,
        'Visibility(mi)': visibility.value,
        'Wind_Speed(mph)': wind.value,
        'Weather_Condition': int(weather_code.value),
        'Junction': int(junction.value),
        'Traffic_Signal': int(traffic_signal.value),
        'Crossing': int(crossing.value),
        'Sunrise_Sunset': int(sunrise.value),
        'Hour': int(hour.value),
        'Day_of_Week': int(day.value),
        'Is_Weekend': int(day.value in [5, 6])
    }])
    
    # Get raw probabilities from the model
    raw_probs = model.predict_proba(input_data)[0]
    
    # Apply our custom threshold weights!
    adjusted_probs = raw_probs * threshold_weights
    
    # The predicted class is the one with the highest adjusted probability
    pred_index = np.argmax(adjusted_probs)
    pred_class = pred_index + 1
    
    # Update UI
    result_label.set_text(f"Predicted Severity: {pred_class}")
    result_label.classes(replace=f"text-4xl font-bold {'text-red-500' if pred_class >= 3 else 'text-green-500'}")
    
    # Update probability bars to show the raw (true) probabilities
    for i in range(4):
        prob_labels[i].set_text(f"Severity {i+1}: {raw_probs[i]*100:.1f}%")

# --- UI LAYOUT ---
ui.dark_mode().enable()

with ui.header().classes('items-center justify-between'):
    ui.label('🚗 US Accident Severity Predictor').classes('text-2xl font-bold')
    ui.label('AI & Expert Systems Project').classes('text-sm text-gray-400')

with ui.row().classes('w-full justify-center gap-8 p-8'):
    
    # LEFT PANEL: Inputs
    with ui.card().classes('w-1/3'):
        ui.label('Crash Conditions').classes('text-xl font-semibold mb-4')
        
        temp = ui.slider(min=-20, max=120, value=70).props('label-always').classes('w-full')
        ui.label('Temperature (F)')
        
        humidity = ui.slider(min=0, max=100, value=50).props('label-always').classes('w-full')
        ui.label('Humidity (%)')
        
        visibility = ui.slider(min=0, max=10, value=10).props('label-always').classes('w-full')
        ui.label('Visibility (mi)')
        
        wind = ui.slider(min=0, max=50, value=5).props('label-always').classes('w-full')
        ui.label('Wind Speed (mph)')
        
        weather_code = ui.number(label='Weather Code (0=Clear, 1=Rain)', value=0)
        
        hour = ui.slider(min=0, max=23, value=12).props('label-always').classes('w-full')
        ui.label('Hour of Day (0-23)')
        
        day = ui.slider(min=0, max=6, value=0).props('label-always').classes('w-full')
        ui.label('Day of Week (0=Mon, 6=Sun)')
        
        with ui.row().classes('w-full justify-between mt-4'):
            junction = ui.checkbox('Junction')
            traffic_signal = ui.checkbox('Traffic Signal')
            crossing = ui.checkbox('Crossing')
            
        sunrise = ui.checkbox('Daytime (Sunrise/Sunset)', value=True)
        
        ui.button('Predict Severity', on_click=predict_severity).classes('w-full mt-6 bg-blue-600')

    # RIGHT PANEL: Outputs
    with ui.card().classes('w-1/3 flex flex-col items-center justify-center'):
        ui.label('Prediction Result').classes('text-xl font-semibold mb-8')
        
        result_label = ui.label('Awaiting Input...').classes('text-4xl font-bold text-gray-400')
        
        ui.separator().classes('my-8')
        
        ui.label('Raw Probability Distribution').classes('text-lg mb-4')
        
        prob_labels = []
        for i in range(4):
            lbl = ui.label(f"Severity {i+1}: 0.0%").classes('text-md w-full text-left')
            prob_labels.append(lbl)

ui.run(title="Accident Severity App", port=8080, reload=False)