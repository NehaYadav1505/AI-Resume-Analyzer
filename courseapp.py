import streamlit as st
import pickle
import pandas as pd

# Load data
try:
    similarity = pickle.load(open('models/similarity.pkl', 'rb'))
    courses_df = pickle.load(open('models/courses.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure all .pkl files exist in the 'models' directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Extract course names
course_names = courses_df['course_name'].values.tolist()

# Recommend function
def recommend(course_name):
    if course_name not in courses_df['course_name'].values:
        return []

    try:
        index = courses_df[courses_df['course_name'] == course_name].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_courses = []
        for i in distances[1:7]:  # Get top 6 excluding self
            name = courses_df.iloc[i[0]]['course_name']
            url = courses_df.iloc[i[0]]['course_url']
            recommended_courses.append({'name': name, 'url': url})
        return recommended_courses
    except Exception as e:
        st.warning(f"Error generating recommendations: {e}")
        return []

# Streamlit UI
st.title("📚 Course Recommendation System")
st.write("Select a course to see similar recommended courses.")

selected_course = st.selectbox("Choose a course:", [""] + course_names)

if selected_course:
    recommendations = recommend(selected_course)
    st.subheader(f"Top Recommendations for: *{selected_course}*")
    if recommendations:
        for course in recommendations:
            st.markdown(f"- [{course['name']}]({course['url']})")
    else:
        st.info("No recommendations found.")
