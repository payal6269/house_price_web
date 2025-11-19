from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# -----------------------------
# Sample dataset
# -----------------------------
data = {
    'Area': [1000, 1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [2, 3, 3, 4, 4, 5],
    'Washroom': ['One', 'Attached', 'One', 'Attached', 'Attached', 'Attached'],
    'DrawingRoom': [1, 1, 0, 1, 1, 1],
    'DiningArea': [1, 1, 0, 1, 1, 1],
    'Location': ['Patiala', 'Chandigarh', 'Amritsar', 'Jaipur', 'Jaisalmer', 'Delhi'],
    'Price': [50000, 75000, 80000, 120000, 150000, 180000]
}

df = pd.DataFrame(data)

# -----------------------------
# Preprocessing
# -----------------------------
df = pd.get_dummies(df, columns=['Location', 'Washroom'], drop_first=True)
X = df.drop('Price', axis=1)
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# Save feature columns to match during prediction
feature_columns = X.columns

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    price = None
    if request.method == "POST":
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        washroom = request.form["washroom"]
        drawing_room = int(request.form.get("drawing_room", 0))
        dining_area = int(request.form.get("dining_area", 0))
        location = request.form["location"]

        # Prepare prediction row with all features
        new_data = {
            'Area': area,
            'Bedrooms': bedrooms,
            'DrawingRoom': drawing_room,
            'DiningArea': dining_area
        }

        # Add Location columns
        for col in [c for c in feature_columns if c.startswith('Location_')]:
            new_data[col] = 1 if col == f'Location_{location}' else 0

        # Add Washroom columns
        for col in [c for c in feature_columns if c.startswith('Washroom_')]:
            new_data[col] = 1 if col == f'Washroom_{washroom}' else 0

        # Convert to DataFrame in correct column order
        new_df = pd.DataFrame([new_data], columns=feature_columns)

        price = model.predict(new_df)[0]

    return render_template("index.html", price=price)

if __name__ == "__main__":
    app.run(debug=True)
