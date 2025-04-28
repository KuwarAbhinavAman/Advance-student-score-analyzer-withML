Advance Student Score Analyzer with ML
Welcome to the Advance Student Score Analyzer with ML, a Streamlit-based web application designed to predict student performance, analyze trends, and provide personalized recommendations using machine learning. This tool helps educators and students track academic progress and identify areas for improvement.
Overview
The Advance Student Score Analyzer uses a RandomForestRegressor model to predict future academic scores based on historical data, attendance, literacy skills, and other metrics. It provides interactive visualizations, actionable recommendations, and downloadable reports to support educational decision-making.

Features:

Upload student data via CSV files.
Analyze performance trends with interactive Plotly charts.
Predict future scores using a machine learning model.
Generate personalized recommendations for attendance, literacy, and performance.
Export analysis reports as CSV files.
Add new students and retrain the model dynamically.


Tech Stack:

Frontend: Streamlit
Backend: Python
Machine Learning: Scikit-learn
Visualization: Plotly
Data Handling: Pandas, NumPy



Live Demo
The app is hosted on Streamlit Community Cloud and can be accessed here:
🔗 Advance Student Score Analyzer with ML
Prerequisites
To run the app locally, ensure you have the following installed:

Python 3.7 or higher
pip (Python package manager)

Installation

Clone the Repository:
git clone https://github.com/KuwarAbhinavAman/Advance-student-score-analyzer-withML.git
cd Advance-student-score-analyzer-withML


Install Dependencies:Create a virtual environment (recommended) and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

The requirements.txt file includes:
pandas
numpy
scikit-learn
streamlit
plotly


Prepare Sample Data:

Create a CSV file (e.g., student_data.csv) with the following structure:Student_Name,Previous Scores,Current Percentage,Attendance,Reading_Score,Writing_Score,Grade
John Doe,"75, 80, 78",82.5,92.0,7.5,8.0,B
Jane Smith,"85, 88, 90",91.0,98.0,9.0,9.5,A


Place the file in the project directory or upload it via the app interface.



Usage

Run the App Locally:
streamlit run main.py

Open your browser and go to http://localhost:8501.

Interact with the App:

Data Management: Upload a student database (CSV) using the sidebar and specify an output file (e.g., student_data.csv).
Student Selection: Choose an existing student or add a new one.
Analysis: Input the current term percentage and click "Analyze Performance" to view trends, predictions, and recommendations.
Download: Export the analysis report as a CSV file.


Sample Workflow:

Upload student_data.csv.
Select "John Doe" and enter the current percentage (e.g., 82.5).
Click "Analyze Performance" to see the report, including charts and recommendations.
Download the report as a CSV file.



Deployment
This app is deployed on Streamlit Community Cloud (free tier), making it accessible via a public URL.
Deployed URL

🔗 https://advance-student-score-analyzer-withml.streamlit.app

Deployment Steps

The code is hosted on GitHub: KuwarAbhinavAman/Advance-student-score-analyzer-withML.
Streamlit Community Cloud pulls the code from the main branch and deploys it using main.py as the entry point.
Dependencies are installed automatically from requirements.txt.

Notes for Deployment

The app is public by default. To restrict access (e.g., for student data privacy), set it to private in the Streamlit dashboard and invite specific viewers.
The free tier has resource limits (e.g., 1GB memory). For larger datasets, consider optimizing the app or upgrading to a paid plan.

Project Structure
Advance-student-score-analyzer-withML/
├── main.py               # Main Streamlit app code
├── requirements.txt      # List of dependencies
├── student_data.csv      # Sample student data (optional)
├── README.md             # Project documentation

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch:git checkout -b feature/your-feature-name


Make your changes and commit:git commit -m "Add your message here"


Push to your fork:git push origin feature/your-feature-name


Open a pull request on the GitHub repository.

License
This project is licensed under the MIT License. See the LICENSE file for details (add a LICENSE file if not already present).
Contact
For questions or support, contact KuwarAbhinavAman or open an issue on the GitHub repository.
Acknowledgments

Built with Streamlit, a powerful framework for data apps.
Thanks to the open-source community for tools like Pandas, Scikit-learn, and Plotly.


Last updated: April 28, 2025
