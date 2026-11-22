import gradio as gr
import joblib
import pandas as pd

model = joblib.load("model.joblib")

# Must match the training column order exactly
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "type_white",
]


def predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_so2, total_so2, density, ph, sulphates,
            alcohol, wine_type):
    row = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_so2, total_so2, density, ph, sulphates, alcohol,
        1 if wine_type == "white" else 0,
    ]]
    X = pd.DataFrame(row, columns=FEATURES)
    score = float(model.predict(X)[0])
    score = max(0.0, min(10.0, score))
    return f"Predicted quality: {score:.1f} / 10  (typical error about ±0.6)"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(3.8, 15.9, value=7.0, label="Fixed acidity"),
        gr.Slider(0.08, 1.6, value=0.3, label="Volatile acidity"),
        gr.Slider(0.0, 1.7, value=0.3, label="Citric acid"),
        gr.Slider(0.6, 66.0, value=3.0, label="Residual sugar"),
        gr.Slider(0.009, 0.62, value=0.047, label="Chlorides"),
        gr.Slider(1, 290, value=29, label="Free sulfur dioxide"),
        gr.Slider(6, 440, value=118, label="Total sulfur dioxide"),
        gr.Slider(0.987, 1.04, value=0.9949, label="Density"),
        gr.Slider(2.7, 4.0, value=3.2, label="pH"),
        gr.Slider(0.22, 2.0, value=0.5, label="Sulphates"),
        gr.Slider(8.0, 15.0, value=10.3, label="Alcohol (%)"),
        gr.Radio(["red", "white"], value="white", label="Wine type"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Wine Quality Predictor",
    description=(
        "Enter the chemical measurements of a Vinho Verde wine to get a "
        "predicted quality score (0-10). Random Forest model trained on the "
        "UCI Wine Quality dataset (Cortez et al., 2009)."
    ),
)

if __name__ == "__main__":
    demo.launch()
