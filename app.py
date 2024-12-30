from flask import Flask, request, render_template, jsonify ,redirect,session,url_for,flash
import pandas as pd
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from langchain.prompts import PromptTemplate
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(128)

db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')

# Load TF-IDF Vectorizers
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def connect_to_db():
    """Establish a connection to the MySQL database."""
    return mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )

def fetch_data(query):
    """Fetch data from the MySQL database."""
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(result)

# Fetch job data once at the start
data_j = fetch_data("SELECT * FROM job_descriptions")
data_j['combined_text'] = data_j['Job_Description'].fillna('') + ' ' + data_j['skills'].fillna('') + ' ' + data_j['Responsibilities'].fillna('')
tfidf_jobs = tfidf_vectorizer.transform(data_j['combined_text'])

# Fetch course data once at the start
data_c = fetch_data("SELECT title AS 'Title', short_intro AS 'Short Intro', url AS 'URL', skills AS 'Skills', site AS 'Site' FROM Courses")
data_c['combined_text'] = data_c['Short Intro'].fillna('') + ' ' + data_c['Skills'].fillna('')
tfidf_courses = tfidf_vectorizer.transform(data_c['combined_text'])

def get_user_profile(name, education, skills, interests):
    """Create a user profile string."""
    return f"{name} {education} {skills} {interests}"

def recommend_jobs(user_profile, top_n=5):
    user_profile_vector = tfidf_vectorizer.transform([user_profile])
    similarity_scores = cosine_similarity(user_profile_vector, tfidf_jobs)
    sorted_scores = sorted(enumerate(similarity_scores[0]), key=lambda x: x[1], reverse=True)
    top_jobs_idx = [i[0] for i in sorted_scores[:top_n]]
    top_jobs = data_j.iloc[top_jobs_idx]
    top_jobs = top_jobs.drop_duplicates(subset='Qualifications',keep='first')
    top_jobs = top_jobs.drop_duplicates(subset='Company',keep='first')
    top_jobs = top_jobs.drop_duplicates(subset='location',keep='first')
    top_jobs = top_jobs.drop_duplicates(subset='Country',keep='first')
    top_jobs = top_jobs.drop_duplicates(subset='Role',keep='first')
    return top_jobs[['Job_Title','Role','Job_Description','skills','Responsibilities']].to_dict(orient='records')

def get_google_ai_response(prompt):
    """Get a response from Google Generative AI."""
    context = (
        "EduQuest is an educational app. The founder of EduQuest is Bharath Kumar. "
        "I am an AI assistant created by EduQuest to assist with educational queries. "
        "If asked 'Who created you?' or 'Who is your creator?', the answer should be 'EduQuest'. "
        "If asked 'Who are you?', the answer should be 'I am an AI assistant created by EduQuest.' "
        "If asked 'Who is your founder?', the answer should be 'The founder of EduQuest is Bharath Kumar.' "
        "If asked 'How are you related to EduQuest?', the answer should be 'I am an AI assistant developed to support EduQuest’s educational services.' "
        "If asked 'List me the recent course doubts?', provide information on recent common questions related to courses at EduQuest. "
        "If asked 'Is EduQuest available?' or 'Can you provide details about EduQuest?', respond with 'The product is still in development and yet to be released.'"
        "If asked 'Can you play games?' or any related question, respond with 'I am a text-based AI and do not have the capability to play games. My function is to assist with educational queries and provide information.'"
        "If asked 'Can you answer questions' or any related question, respond with 'Yes I can answer any technical or non technical questions.Feel Free to Ask me anything!!!'"
    )
    # Get the API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY_1")

    # Configure the generative AI with the API key
    genai.configure(api_key=api_key)

    # Initialize the generative model
    model = genai.GenerativeModel('gemini-1.5-flash')

    chat = model.start_chat(history=[])

    chat.send_message(context)

    response = chat.send_message(prompt, stream=False)

    return response.text


def get_user_by_username_or_email(username_email):
    conn = connect_to_db()  
    cursor = conn.cursor(dictionary=True)  
    query = """
    SELECT * FROM user_profiles WHERE username = %s OR email = %s
    """
    try:
        cursor.execute(query, (username_email, username_email))
        user = cursor.fetchone()  
        
        return user  
    
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        cursor.close()
        conn.close()

def get_skills_interests(username):
    conn = connect_to_db()
    cursor = conn.cursor()
    
    query = "SELECT skills, interests FROM user_profiles WHERE username = %s"
    cursor.execute(query, (username,))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if result:
        return result  
    else:
        return None, None

def generate_questions(username):
    api_key = os.getenv("GOOGLE_API_KEY_2")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    skills, interests = get_skills_interests(username)

    if skills and interests:
        prompt = f"""
        Generate 10 unique multiple-choice questions based on the following:
        Skills: {skills}
        Each question must have:
        - A clear question statement.
        - Four answer options (a, b, c, d).
        - Only one correct answer.
        Format as: question|option a|option b|option c|option d|correct answer
        """

        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        response_text = response.text.strip()

        questions_data = response_text.splitlines()
        valid_questions = []

        for entry in questions_data:
            parts = entry.split('|')
            if len(parts) == 6:
                question, option_a, option_b, option_c, option_d, correct_answer = parts
                valid_questions.append(f"{question}|{option_a}|{option_b}|{option_c}|{option_d}")
            else:
                print(f"Invalid format: {entry}")

        return "\n".join(valid_questions)
    return "User profile not found or missing skills and interests."
        
def fetch_generated_questions(username):
    conn = connect_to_db()
    
    try:
        cursor = conn.cursor()
        query = "SELECT question, correct_answer FROM generated_questions WHERE username = %s"
        cursor.execute(query, (username,))
        result = cursor.fetchall()
        questions = [{'question': row[0], 'correct_answer': row[1]} for row in result]
        
    finally:
        cursor.close()
        conn.close()
    
    return questions

def parse_generated_questions(response_text):

    questions = []
    # Split response text by lines to separate each question
    lines = response_text.strip().split('\n')
    
    # Iterate over each line to extract the question and answers
    for line in lines:
        # Assuming each line follows the format "Question | Option1 | Option2 | Option3 | CorrectAnswer"
        parts = line.split('|')
        if len(parts) == 5:  # Expecting exactly 5 parts (1 question and 4 options)
            question = parts[0].strip()
            correct_answer = parts[-1].strip()  # Assuming the correct answer is the last part
            questions.append((question, correct_answer))
    
    return questions

def insert_or_update_generated_questions(username,questions):
    conn = connect_to_db()
    cursor=conn.cursor()
    try:
        questions = parse_generated_questions(questions)

        if not questions:
            print("No questions generated.")
            return
        
        cursor = conn.cursor()
        check_query = "SELECT COUNT(*) FROM generated_questions WHERE username = %s"
        cursor.execute(check_query, (username,))
        result = cursor.fetchone()

        if result[0] > 0:
            delete_query = "DELETE FROM generated_questions WHERE username = %s"
            cursor.execute(delete_query, (username,))

        insert_query = "INSERT INTO generated_questions (username, question, correct_answer) VALUES (%s, %s, %s)"
        for question, correct_answer in questions:
            cursor.execute(insert_query, (username, question, correct_answer))

        conn.commit()

    finally:
        cursor.close()
        conn.close()

def generate_roadmap(job_title):
    api_key = os.getenv("GOOGLE_API_KEY_3")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(history=[])

    prompt = f"""
    Create a comprehensive career roadmap for becoming a {job_title}. The roadmap should include key skills to acquire and recommend relevant courses at each timeline point. Structure the roadmap in phases:

    1. **Foundation Stage** : Focus on learning basic and fundamental skills required to become a {job_title}. Mention key introductory courses and beginner skills.
    
    2. **Intermediate Stage** : Develop intermediate skills and pursue certifications. Provide practical project ideas and relevant courses to enhance knowledge in this phase.
    
    3. **Advanced Stage** : Suggest advanced courses, certifications, and specialized skills. Include ways to apply this knowledge in real-world scenarios, build a portfolio, and gain hands-on experience.
    
    4. **Mastery Stage** : Achieve mastery in key areas. Recommend expert-level courses and skills to help the user excel and become an expert {job_title}.
    
    For each phase, suggest skills, tools, certifications, and specific course names relevant to becoming a {job_title}. Include recommendations for both online platforms and real-world application strategies.

    Also give me the hyperlinks of the courses
    
    Also give me the Qualifications needed in each stage
    """
    
    response = chat.send_message(prompt)
    return response

def fetch_user_profile():
    conn = connect_to_db()
    cursor = conn.cursor()
    query = "SELECT interests, location FROM user_profiles"
    cursor.execute(query)
    return cursor.fetchone()

# Function to scrape Google Maps
def scrape_google_maps(interest, location):
    search_query = f"{interest} workshops and classes in {location}".replace(' ', '+')
    url = f'https://www.google.com/maps/search/{search_query}/'

    # Configure headless browser options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # Initialize Chrome WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)

    # Wait for content to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    main_content = driver.find_element(By.TAG_NAME, 'body').text

    # Clean the extracted text
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\n]', '', main_content)
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()
    scraped_text = "\n".join(f"- {line.strip()}" for line in cleaned_text.splitlines() if line.strip())

    driver.quit()
    return scraped_text


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_email = request.form['username_email']
        password = request.form['password']

        # Establish database connection
        conn = connect_to_db()
        cursor = conn.cursor()

        # Check if user exists using either username or email
        query = "SELECT username, password FROM user_profiles WHERE username=%s OR email=%s"
        cursor.execute(query, (username_email, username_email))
        user = cursor.fetchone()

        if user is None:
            error = "User not registered. Please register first."
            return render_template('login.html', error=error)

        # Check if the password matches
        if check_password_hash(user[1], password):
            # Login successful
            session['username'] = user[0]  # Set the username in session
            return redirect(url_for('home'))
        else:
            error = "Invalid password. Please try again."
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        try:
            conn = connect_to_db()
            cur = conn.cursor()

            # Insert new user
            cur.execute("INSERT INTO user_profiles (email, username, password) VALUES (%s, %s, %s)", (email, username, password))
            conn.commit()
            return redirect(url_for('login'))

        except mysql.connector.Error as err:
            # Check for duplicate entry error
            if err.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                return render_template('register.html', error="Username or email already taken.")
            else:
                print(f"Error: {err}")
                return render_template('register.html', error="Registration failed, please try again.")

        finally:
            cur.close()
            conn.close()

    return render_template('register.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username_email = request.form['username_email']

        # Connect to the database and check if the user exists
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM user_profiles WHERE username=%s OR email=%s", (username_email, username_email))
        user = cursor.fetchone()

        if user:
            # Optionally send an email with a reset link (we'll skip that for simplicity)
            return redirect(url_for('reset_password', username_email=username_email))
        else:
            error = "No user found with that email or username."
            return render_template('forgot_password.html', error=error)

    return render_template('forgot_password.html')

@app.route('/reset_password/<username_email>', methods=['GET', 'POST'])
def reset_password(username_email):
    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            error = "Passwords do not match."
            return render_template('reset_password.html', error=error, username_email=username_email)

        # Hash the new password
        hashed_password = generate_password_hash(new_password)

        # Update the password in the database
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE user_profiles SET password = %s WHERE username = %s OR email = %s", 
                       (hashed_password, username_email, username_email))
        conn.commit()
        cursor.close()
        conn.close()

        flash('Password reset successful. You can now log in with the new password.')
        return redirect(url_for('login'))

    return render_template('reset_password.html', username_email=username_email)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/courses')
def courses():
    return render_template('courses.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/profile')
def profile():
    if 'username' in session:
        username = session['username']
        conn = connect_to_db()
        cursor = conn.cursor(dictionary=True)

        # Fetch user profile data from the database
        cursor.execute("SELECT * FROM user_profiles WHERE username = %s", (username,))
        user_profile = cursor.fetchone()

        cursor.close()
        conn.close()

        # Render the profile page with user data
        return render_template('profile.html', user_profile=user_profile)
    else:
        return redirect(url_for('login'))

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/roadmap')
def roadmap():
    return render_template('roadmap.html')

@app.route('/location', methods=['GET'])
def location():
    return render_template('location.html')

    
@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle job and course recommendations based on the user profile."""
    name = request.form.get('name')
    education = request.form.get('education')
    skills = request.form.get('skills')
    skills_other = request.form.get('skills-other-text', '')
    interests = request.form.get('interests')
    interests_other = request.form.get('interests-other-text', '')

    # Append "Other" text to the main field if "Other" was selected
    if skills == 'Other':
        skills = skills_other
    if interests == 'Other':
        interests = interests_other

    user_profile = get_user_profile(name, education, skills, interests)
    recommended_jobs = recommend_jobs(user_profile)

    return jsonify({'status': 'success', 'jobs': recommended_jobs})

@app.route('/save_profile', methods=['POST'])
def save_profile():
    if 'username' in session:
        username = session['username']  # Retrieve the logged-in user's username

        # Get user input from the request
        name = request.form.get('name')
        education = request.form.get('education')
        skills = request.form.get('skills')
        skills_other = request.form.get('skills-other-text', '')
        interests = request.form.get('interests')
        interests_other = request.form.get('interests-other-text', '')
        location = request.form.get('location')

        # Append "Other" text to the main field if "Other" was selected
        if skills == 'Other':
            skills = skills_other
        if interests == 'Other':
            interests = interests_other

        try:
            # Connect to the database
            conn = connect_to_db()
            cursor = conn.cursor()

            # Update the user profile with the new details
            query = """
            UPDATE user_profiles 
            SET name = %s, education = %s, skills = %s, interests = %s, location = %s
            WHERE username = %s
            """
            cursor.execute(query, (name, education, skills, interests, location, username))

            conn.commit()  # Commit the changes
            cursor.close()  # Close the cursor
            conn.close()  # Close the database connection

            return jsonify({'status': 'success', 'message': 'Profile saved successfully.'})

        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    else:
        return jsonify({'status': 'error', 'message': 'User not logged in.'})


@app.route('/search_courses')
def search_courses():
    """Search for courses based on the query parameter."""
    query = request.args.get('query', '')
    
    if not query:
        
        return jsonify({'courses': []})
    
    
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)

    
    sql_query = """
        SELECT title AS 'Title', short_intro AS 'Short Intro', url AS 'URL', skills AS 'Skills', site AS 'Site' 
        FROM Courses 
        WHERE title LIKE %s OR short_intro LIKE %s
    """
    search_term = f"%{query}%"  
    cursor.execute(sql_query, (search_term, search_term))
    
    result = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return jsonify({'courses': result})

@app.route('/get_chatbot_response', methods=['POST'])
def get_chatbot_response():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        response = get_google_ai_response(prompt)  
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': 'Sorry, there was an error processing your request due to content safety or other concerns.'}), 400

@app.route('/generate-questions', methods=['GET'])
def generate_questions_endpoint():
    # Get the username from session
    username = session.get('username')  # Assume username is stored in the session
    if username:
        questions = generate_questions(username)
        insert_or_update_generated_questions(username, questions)
        return jsonify({'questions': questions})
    else:
        return jsonify({'error': 'User not logged in.'}), 401

@app.route('/submit-test', methods=['POST'])
def submit_test():
    if 'username' not in session:
        return jsonify({'error': 'User not logged in.'}), 401

    username = session['username']
    user_answers = request.form  # The submitted answers from the frontend

    # Fetch the generated questions from the database
    questions = fetch_generated_questions(username)  # Assume this function gets the questions and correct answers from the DB
    
    score = 0
    total_questions = len(questions)

    # Example: questions are stored as list of dictionaries with 'question' and 'correct_answer'
    for i, question in enumerate(questions):
        q_key = f'question-{i}'  # This is the form field name sent from frontend
        user_answer = user_answers.get(q_key)

        if user_answer and user_answer == question['correct_answer']:
            score += 1

    return jsonify({'score': score, 'total': total_questions})

@app.route('/generate_roadmap', methods=['POST'])
def generate_roadmap_route():
    data = request.json
    job_title = data.get('job_title')

    # Generate the roadmap using the model
    roadmap_response = generate_roadmap(job_title)

    roadmap_text = roadmap_response.text  

    # Define the phases for splitting the response
    phases = ["Foundation Stage", "Intermediate Stage", "Advanced Stage", "Mastery Stage"]
    phase_split_data = {phase: "" for phase in phases}  

    # Current phase tracker
    current_phase = None

    # Split the text by lines
    roadmap_lines = roadmap_text.split("\n")

    # Process each line to classify it under the right phase
    for line in roadmap_lines:
        # Check if the line starts a new phase
        if any(phase in line for phase in phases):
            current_phase = next(phase for phase in phases if phase in line)
        if current_phase:
            phase_split_data[current_phase] += line + "\n"

    # Return the structured response with each phase
    return jsonify({
        "roadmap": {
            "foundation": phase_split_data["Foundation Stage"].strip(),
            "intermediate": phase_split_data["Intermediate Stage"].strip(),
            "advanced": phase_split_data["Advanced Stage"].strip(),
            "mastery": phase_split_data["Mastery Stage"].strip()
        }
    })

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    # Fetch user data from the database
    user_data = fetch_user_profile()
    if not user_data:
        return jsonify({"error": "User data not found."}), 404

    interests, location = user_data

    # Scrape Google Maps for relevant data
    scraped_data = scrape_google_maps(interests, location)

    # Configure generative AI
    api_key = os.getenv("GOOGLE_API_KEY_1")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(history=[])

    # Prepare prompt for generative AI
    final_prompt = (
        f"Provide detailed information on workshops and classes conducted, including locations, addresses, "
        f"and names based on the following data:\n\n{scraped_data}\n"
        "Exclude websites and do not add bullet points; format in a table."
    )
    response = chat.send_message(final_prompt)

    return jsonify({"recommendations": response.text})

if __name__ == '__main__':
    app.run(debug=True)