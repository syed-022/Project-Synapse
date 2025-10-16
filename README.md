Project Synapse: An AI-Based Intelligent System for Data-Driven Decision Making

Project Synapse is a full-stack web application designed to empower non-technical domain experts with the tools of modern machine learning. It transforms raw, tabular data into actionable, trustworthy, and explainable decisions through a seamless and interactive user experience.

This platform bridges the gap between complex AI models and human intuition, moving beyond simple predictions to provide deep, understandable insights.

Live Demo
    Click here to see the live application <!-- IMPORTANT: Replace the above URL with your actual Render URL once it's deployed! --><!--
    A GIF showing the workflow: 1. Upload CSV -> 2. View model results -> 3. Adjust sliders on simulator -> 4. See new prediction. -->
    
Core Features
    * Data-to-Decision Pipeline: Upload any structured CSV dataset and automatically train, benchmark, and select the best-performing machine learning model for your specific problem.
    * Interactive "What-If" Simulator: Dynamically adjust input variables and receive real-time predictions, turning the model into an exploratory tool for decision-making.
    * Explainable AI (XAI) Core: Demystifies the "black box." Every prediction is accompanied by a clear visualization showing the exact factors that influenced the outcome, powered by SHAP (SHapley Additive exPlanations).
    * Conversational AI Assistant: A built-in chatbot that can explain technical concepts, clarify results, and provide suggestions for improving model performance, making the platform accessible to users of all skill levels.
    
Tech Stack
Component        Technology
Frontend         React.js, Tailwind CSS
Backend          Python, FastAPI
ML/AI            Pandas, Scikit-learn, XGBoost, SHAP
Deployment       Docker, Render

Getting Started: Running Locally
  To run this project on your local machine, please follow these steps.
  Prerequisites:
    Python 3.9+
    Node.js and npm
    Git
    
  STEP 1 : Clone the repository:
    git clone [https://github.com/syed-022/Project-Synapse.git](https://github.com/your-username/Project-Synapse.git)
    cd Project-Synapse

  STEP 2 : Set up and run the Backend:
    # Navigate to the backend directory 
    cd backend

    # Create a virtual environment and activate it
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the required packages
    pip install -r requirements.txt

    # Run the FastAPI server
    uvicorn main:app --reload
The backend will now be running on http://localhost:8000.

  STEP 3 : Set up and run the Frontend:
    # Navigate to the frontend directory from the root folder
    cd frontend

    # Install the required npm packages
    npm install

    # Run the React development server
    npm start
The frontend will now be running on http://localhost:3000 and will be connected to your local backend.

Deployment :
This project is configured for easy, automated deployment on Render.
1.Push to GitHub: Ensure all your code is pushed to a GitHub repository.
2.Create a Blueprint Service: Connect your repository to Render and create a new Blueprint service.
3.Automatic Deployment: Render will automatically find the "render.yaml" file, build the Docker images for both the frontend and backend, and deploy them. Any future push to your "main" branch will trigger a new deployment.
