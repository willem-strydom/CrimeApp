import os
import shutil

def setup_flask_app():
    """
    Setup the Flask application by creating necessary directories and files.
    """
    print("Setting up Flask application for Hawkes Process Crime Prediction Portfolio...")
    
    # Create main directories if they don't exist
    dirs = ['templates', 'static', 'static/css', 'static/js', 'static/img']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    
    # Create app.py if it doesn't exist (main Flask application)
    if not os.path.exists('app.py'):
        # You'll need to create the app.py file manually or copy it from your provided code
        print("Please create app.py using the Flask application code provided.")
    
    # Create HTML templates
    templates = {
        'index.html': 'templates/index.html',
        'methodology.html': 'templates/methodology.html',
        'about.html': 'templates/about.html'
    }
    
    for template_name, template_path in templates.items():
        if not os.path.exists(template_path):
            print(f"Please create {template_path} using the HTML template code provided.")
    
    print("\nSetup complete! Next steps:")
    print("1. Make sure you've placed all HTML templates in the 'templates' directory")
    print("2. Run the Flask app with: python app.py")
    print("3. Visit http://127.0.0.1:5000/ in your browser to view your portfolio")

def create_requirements_file():
    """
    Create a requirements.txt file for the project.
    """
    requirements = [
        "Flask==2.2.3",
        "numpy==1.24.2",
        "pandas==1.5.3",
        "matplotlib==3.7.1",
        "scikit-learn==1.2.2"
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    print("Created requirements.txt")
    print("To install dependencies, run: pip install -r requirements.txt")

def create_readme():
    """
    Create a README.md file for the project.
    """
    readme_content = """# Hawkes Process Crime Prediction Portfolio

A web application showcasing the implementation of Hawkes process models for crime prediction.

## Overview

This portfolio project demonstrates the application of spatial-temporal point process models, 
specifically Hawkes processes, for predicting crime patterns. The project combines:

- Kernel Density Estimation (KDE) for spatial modeling
- SARIMA time series forecasting for temporal prediction
- Self-exciting Hawkes processes to capture how crime events trigger nearby future incidents

## Features

- Interactive visualizations of crime density patterns
- Time series forecasting of crime incidents
- Comprehensive methodology explanation
- Detailed model evaluation metrics

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and navigate to http://127.0.0.1:5000/

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## License

MIT License

## Contact

Your Name - your.email@example.com
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("Created README.md")

def create_sample_resume():
    """
    Create a dummy resume.pdf file for the portfolio.
    This is just a placeholder - you should replace it with your actual resume.
    """
    resume_content = """
    This is a placeholder for your actual resume.
    Please replace this file with your real resume in PDF format.
    """
    
    with open('static/resume.pdf', 'w') as f:
        f.write(resume_content)
    
    print("Created placeholder resume file at static/resume.pdf")

def create_gitignore():
    """
    Create a .gitignore file for the project.
    """
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Flask
instance/
.webassets-cache

# IDEs and editors
.idea/
.vscode/
*.swp
*.swo
*~

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("Created .gitignore file")

def setup_deployment_files():
    """
    Create files needed for deployment to platforms like Heroku or Vercel.
    """
    # Procfile for Heroku
    with open('Procfile', 'w') as f:
        f.write('web: gunicorn app:app')
    
    print("Created Procfile for Heroku deployment")
    
    # Add gunicorn to requirements
    with open('requirements.txt', 'a') as f:
        f.write('\ngunicorn==20.1.0')
    
    print("Added gunicorn to requirements.txt")

if __name__ == "__main__":
    setup_flask_app()
    create_requirements_file()
    create_readme()
    create_sample_resume()
    create_gitignore()
    setup_deployment_files()
    
    print("\nAll setup tasks completed!")
    print("You can now customize the templates and code to fit your specific needs.")
    print("To run the application locally: python app.py")