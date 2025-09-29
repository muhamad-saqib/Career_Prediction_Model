# Career Prediction Web App

This is a Career Prediction Web Application built with Python, Flask, and Machine Learning. 
The app predicts the most suitable career paths for users based on their academic scores, skills, and interests.

# Features:
- Input user data:
  - Matric Percentage
  - Intermediate Percentage
  - Skills (multi-select)
  - Interests (multi-select)
- Machine Learning model predicts career paths.
- User-friendly web interface.
- Fully interactive form for input and result display.

# Project Structure:
career_prediction/
├── app.py                # Flask application
├── model.pkl             # Trained ML model
├── templates/            # HTML templates
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

# Installation:
1. Clone the repository:
   git clone <your-github-repo-link>
   cd career_prediction

2. Create a virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Run the Flask app:
   python app.py

5. Open in browser:
   http://127.0.0.1:5000

# Usage:
- Enter your Matric and Intermediate percentages.
- Select your Skills and Interests from the multi-select options.
- Click Submit to get the predicted career path.

# Deployment:
This project can be deployed on:
- Render
- Heroku

# Dependencies:
- Flask==2.3.3
- pandas==2.1.0
- scikit-learn==1.3.0
- numpy==1.26.0
- gunicorn==22.1.0

# Author
Muhammad Saqib<br>
BSCS-SZABIST'26 | Python Development, Machine Learning<br>

GitHub: @muhamad-saqib

