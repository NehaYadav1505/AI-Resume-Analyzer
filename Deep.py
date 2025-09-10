import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import PyPDF2
import io
import base64
from typing import Union

# Download NLTK data
nltk.download('stopwords')

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("updatedResumeDataSet.csv")
    return df

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Stemming
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    
    return text

# Train model
@st.cache_resource
def train_model(df):
    # Preprocess text
    df['cleaned_resume'] = df['Resume'].apply(preprocess_text)
    
    # Encode categories
    label_encoder = LabelEncoder()
    df['Category_encoded'] = label_encoder.fit_transform(df['Category'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_resume'], 
        df['Category_encoded'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    return pipeline, label_encoder

# Extract text from PDF
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from uploaded file (PDF or TXT)
def extract_text(uploaded_file) -> Union[str, None]:
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    return None

# Define job role skills (expanded version)
JOB_ROLE_SKILLS = {

    
        'Data Science': ['Python', 'Machine Learning', 'Scikit-learn', 'Natural Language Processing', 'Statistics', 
                         'Deep Learning', 'TensorFlow', 'Keras', 'Data Analysis', 'Pandas', 'Numpy', 'SQL', 'Tableau'],
        'HR': ['HR Management', 'Payroll', 'Recruitment', 'Employee Relations', 'Training', 'Performance Management'],
        'Advocate': ['Legal Research', 'Drafting', 'Litigation', 'Contracts', 'Court', 'Legal Writing'],
        'Arts': ['Painting', 'Drawing', 'Graphic Design', 'Craft', 'Illustration', 'Photoshop'],
        'Web Designing': ['HTML', 'CSS', 'JavaScript', 'Bootstrap', 'jQuery', 'Angular', 'React', 'Node.js', 'Express.js', 'Django', 'Flask'],
        'Android Developer': ['Android SDK', 'Kotlin', 'Java', 'Gradle', 'XML', 'Android Studio', 'Firebase', 'REST API', 'Material Design', 'Flutter', 'Dart'],
        'IOS Developer': ['Swift', 'Objective-C', 'Xcode', 'Cocoa Touch', 'UI Kit', 'Core Data', 'REST API', 'MVVM', 'VIPER'],
        'UI/UX Designer': ['Wireframing', 'Prototyping', 'User Research', 'Usability Testing', 'Figma', 'Adobe XD', 'Sketch', 'User Flows', 'Information Architecture', 'Design Systems'],
        'Python Developer': ['Python', 'Django', 'Flask', 'FastAPI', 'REST API', 'SQL Alchemy', 'ORM', 'Celery', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'Google Cloud'],
        'Java Developer': ['Java', 'Spring Boot', 'Hibernate', 'Maven', 'Gradle', 'RESTful API', 'Microservices', 'JPA', 'JUnit', 'Spring Security'],
        'DotNet Developer': ['.NET', 'C#', 'ASP.NET', 'Entity Framework', 'MVC', 'Web API', 'Azure DevOps', 'SQL Server'],
        'Testing': ['Manual Testing', 'Automation Testing', 'Selenium', 'JMeter', 'API Testing', 'Test Plans', 'QA', 'Regression Testing', 'Performance Testing'],
        'DevOps Engineer': ['Docker', 'Kubernetes', 'Jenkins', 'AWS', 'Azure', 'GCP', 'CI/CD', 'Ansible', 'Terraform', 'Shell Scripting', 'Linux', 'Git'],
        'SAP Developer': ['SAP ABAP', 'SAP Fiori', 'OData', 'S/4HANA', 'SAPUI5', 'Web Dynpro', 'Workflow'],
        'Sales': ['CRM', 'Salesforce', 'Negotiation', 'Lead Generation', 'Cold Calling', 'Client Relations', 'Sales Strategy'],
        'PMO': ['Project Management', 'PMO', 'Scrum', 'Agile', 'PRINCE2', 'Risk Management', 'Stakeholder Management'],
        'Business Analyst': ['Business Analysis', 'Requirements Gathering', 'Stakeholder Communication', 'Data Modeling', 'SQL', 'JIRA', 'Confluence'],
        'Electrical Engineering': ['AutoCAD Electrical', 'MATLAB', 'Circuit Design', 'PLC', 'SCADA', 'Power Systems', 'Control Systems'],
        'Mechanical Engineer': ['CAD', 'SolidWorks', 'ANSYS', 'Fluid Dynamics', 'Thermodynamics', 'Manufacturing Processes', 'Material Science'],
        'Health and Fitness': ['Nutrition', 'Exercise Science', 'Personal Training', 'Client Assessment', 'Fitness Program Design'],
        'Database': ['SQL', 'NoSQL', 'MongoDB', 'MySQL', 'PostgreSQL', 'Oracle', 'Database Design', 'Database Administration'],
        'Hadoop': ['Hadoop', 'Spark', 'Hive', 'Pig', 'MapReduce', 'HDFS', 'Kafka', 'Data Warehousing'],
        'Blockchain': ['Blockchain', 'Solidity', 'Ethereum', 'Smart Contracts', 'Cryptography', 'Web3', 'Hyperledger'],
        'ETL Developer': ['ETL', 'SQL', 'Data Warehousing', 'SSIS', 'Informatica', 'Talend', 'Data Governance'],
        'Network Security Engineer': ['Network Security', 'Firewalls', 'Intrusion Detection', 'VPN', 'Cybersecurity', 'Ethical Hacking', 'Network Protocols'],
        'Operations Manager': ['Operations Management', 'Logistics', 'Supply Chain', 'Process Improvement', 'Lean Management', 'Six Sigma', 'Inventory Management'],
        'Civil Engineer': ['AutoCAD Civil 3D', 'Structural Analysis', 'Geotechnical Engineering', 'Construction Management', 'Surveying', 'Revit'],
        'Automation Testing': ['Selenium', 'Test Automation', 'API Testing', 'CI/CD Pipelines', 'JUnit', 'TestNG', 'Cucumber', 'Robot Framework']
    }


    # ... (keep other roles the same or expand them similarly)

def calculate_ats_score(resume_text, predicted_role):
    required_skills = JOB_ROLE_SKILLS.get(predicted_role, [])
    present_skills = []
    
    # Convert to lowercase for case-insensitive matching
    resume_text_lower = resume_text.lower()
    
    for skill in required_skills:
        # Check for both exact match and stemmed version
        skill_variations = [skill.lower(), preprocess_text(skill)]
        if any(variation in resume_text_lower for variation in skill_variations):
            present_skills.append(skill)
    
    if not required_skills:
        return 0, [], []
    
    score = (len(present_skills) / len(required_skills)) * 100
    missing_skills = [skill for skill in required_skills if skill not in present_skills]
    
    return round(score, 2), present_skills, missing_skills

def create_download_link(resume_text, filename="analyzed_resume.txt"):
    """Generate a download link for the analyzed resume"""
    b64 = base64.b64encode(resume_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Analysis Report</a>'
    return href

def main():
    st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        .stButton>button {background-color: #4CAF50; color: white;}
        .stDownloadButton>button {background-color: #2196F3; color: white;}
        .skill-match {color: #4CAF50; font-weight: bold;}
        .skill-missing {color: #F44336; font-weight: bold;}
        .header {color: #1E88E5;}
    </style>
    """, unsafe_allow_html=True)
    
    # Load data and train model
    df = load_data()
    model, label_encoder = train_model(df)
    
    st.title("📄 Resume Analyzer")
    st.markdown("""
    Upload your resume in PDF or TXT format to:
    - Predict the most suitable job role
    - Calculate your ATS (Applicant Tracking System) score
    - See required skills for the role
    - Identify skills you already have vs. missing skills
    """)
    
    # Input options
    input_method = st.radio("Choose input method:", 
                          ("Upload Resume (PDF/TXT)", "Paste Resume Text"),
                          horizontal=True)
    
    resume_text = ""
    if input_method == "Upload Resume (PDF/TXT)":
        uploaded_file = st.file_uploader("Upload your resume:", 
                                       type=["pdf", "txt"],
                                       accept_multiple_files=False,
                                       help="Upload your resume in PDF or TXT format")
        
        if uploaded_file is not None:
            with st.spinner("Extracting text from your resume..."):
                try:
                    resume_text = extract_text(uploaded_file)
                    if not resume_text:
                        st.error("Could not extract text from the file. Please try another file.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    else:
        resume_text = st.text_area("Paste your resume text here:", height=300)
    
    if st.button("Analyze Resume") and resume_text:
        with st.spinner("Analyzing your resume..."):
            # Preprocess and predict
            cleaned_text = preprocess_text(resume_text)
            prediction_encoded = model.predict([cleaned_text])
            predicted_role = label_encoder.inverse_transform(prediction_encoded)[0]
            
            # Calculate ATS score and skills
            ats_score, present_skills, missing_skills = calculate_ats_score(resume_text, predicted_role)
            
            # Display results
            st.success(f"Predicted Job Role: **{predicted_role}**")
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 ATS Score Analysis")
                # Create a gauge chart for ATS score
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(['ATS Score'], [ats_score], 
                       color='#4CAF50' if ats_score > 70 else '#FFC107' if ats_score > 40 else '#F44336')
                ax.set_xlim(0, 100)
                ax.set_title('Your ATS Compatibility Score', pad=15)
                ax.set_xlabel('Score (%)')
                st.pyplot(fig)
                
                st.write(f"Your resume matches **{ats_score}%** of the typical requirements for this role.")
                
                if ats_score > 80:
                    st.success("🎯 Excellent match! Your resume is well-optimized for this role.")
                elif ats_score > 50:
                    st.warning("🔄 Good match, but could be improved with some additional skills.")
                else:
                    st.error("⚠️ Low match. Consider adding more relevant skills for this role.")
            
            with col2:
                st.subheader("📋 Role Requirements")
                st.write(f"**Typical skills for {predicted_role}:**")
                st.write(", ".join(JOB_ROLE_SKILLS.get(predicted_role, [])))
                
                # Show top 5 most important skills
                st.markdown("**Top 5 Critical Skills:**")
                top_skills = JOB_ROLE_SKILLS.get(predicted_role, [])[:5]
                for skill in top_skills:
                    if skill in present_skills:
                        st.markdown(f"- ✅ {skill}")
                    else:
                        st.markdown(f"- ❌ {skill}")
            
            # Skills comparison
            st.subheader("🔍 Skills Analysis")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("### ✅ Skills You Have")
                if present_skills:
                    for skill in present_skills:
                        st.markdown(f"- {skill}")
                    st.metric(label="Skills Match", value=f"{len(present_skills)}/{len(JOB_ROLE_SKILLS.get(predicted_role, []))}")
                else:
                    st.info("No matching skills detected in your resume.")
            
            with col4:
                st.markdown("### ❌ Skills to Improve")
                if missing_skills:
                    for skill in missing_skills:
                        st.markdown(f"- {skill}")
                    st.metric(label="Missing Skills", value=len(missing_skills))
                else:
                    st.success("🌟 Excellent! Your resume contains all typical skills for this role!")
            
            # Tips section
            # st.subheader("💡 Optimization Tips")
            
            # if ats_score < 80:
            #     st.markdown("""
            #     - **Highlight relevant skills** more prominently in your resume
            #     - **Add missing skills** if you have experience with them
            #     - **Use keywords** from the job description
            #     - **Quantify achievements** (e.g., "Improved efficiency by 30%")
            #     - **Keep formatting simple** for ATS readability
            #     - **Include a skills section** with relevant technical skills
            #     """)
                
                # Specific recommendations based on missing skills
                if missing_skills:
                    st.markdown("**Focus on adding these skills:**")
                    cols = st.columns(3)
                    for i, skill in enumerate(missing_skills[:6]):
                        cols[i%3].markdown(f"- {skill}")
            # else:
            #     st.success("Your resume is already well-optimized! Consider these refinements:")
            #     st.markdown("""
            #     - **Tailor it further** for specific job postings
            #     - **Add metrics** to quantify your achievements
            #     - **Update with recent projects** and technologies
            #     """)
            
            # Download analysis report
            st.markdown("---")
            analysis_report = f"""
            Resume Analysis Report
            ---------------------
            Predicted Job Role: {predicted_role}
            ATS Score: {ats_score}%
            
            Skills Match: {len(present_skills)}/{len(JOB_ROLE_SKILLS.get(predicted_role, []))}
            
            Present Skills:
            {', '.join(present_skills)}
            
            Missing Skills:
            {', '.join(missing_skills)}
            
            Recommendations:
            - Focus on adding these skills: {', '.join(missing_skills[:5])}
            - Highlight your experience with: {', '.join(present_skills[:5])}
            """
            
            st.markdown(create_download_link(analysis_report), unsafe_allow_html=True)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>This tool uses machine learning to predict job roles and calculate ATS scores based on resume content.</p>
        <p>Results are estimates based on typical role requirements.</p>
        <p>For best results, ensure your resume is in a readable text format.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()