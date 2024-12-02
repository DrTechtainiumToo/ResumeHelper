# See markdown documents for overview and todolsit

print(f"{'─' * 80}\n\nLOADING...\n")
print("Importing Libraries -> ", end="")

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from collections import Counter
from docx import Document
import requests
from bs4 import BeautifulSoup
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer 
#TODO: import pandas as pd #SWITCH TO GOOGLESHEET OR EXCEL IDK, make gsheet first and backend then transfer to excel

print("Modules Imported\n")

print("Importing NLP Model -> ", end="")
# Global loading for efficiency, If I reload it every time I call a function, I'll waste resources.
logging.basicConfig(level=logging.INFO)
nlp = spacy.load("en_core_web_md")

print("NLP Model Loaded\n")

print("Loading Settings -> ", end="")

#Default Settings
#TODO use configiration file for defailt settings
resume_doc_filepath: str = "resume.docx"
job_post_url: str = None
default_testing: bool = True
    
# BASIC TESTING STUFF
basic_job_description = """We are seeking a highly motivated Data Analyst to join our team. The ideal candidate will have 
a strong background in data analytics, statistical modeling, and visualization. You will be 
responsible for analyzing complex datasets, generating actionable insights, and supporting 
strategic decision-making processes.

Key Responsibilities:
- Analyze large datasets to identify trends and patterns.
- Create interactive dashboards using tools like Tableau or Power BI.
- Collaborate with cross-functional teams to understand business needs.
- Develop predictive models using Python or R.
- Write SQL queries to extract and transform data from relational databases.

Qualifications:
- Bachelor's degree in Statistics, Mathematics, Computer Science, or a related field.
- Proficiency in Python, R, or other statistical programming languages.
- Experience with SQL for data extraction and manipulation.
- Familiarity with Tableau, Power BI, or similar visualization tools.
- Strong communication and presentation skills."""

basic_resume_text = """John Doe
Data Analyst
johndoe@example.com | LinkedIn: linkedin.com/in/johndoe | Phone: (123) 456-7890

Professional Summary:
Detail-oriented data analyst with 3+ years of experience in interpreting and analyzing data 
to drive successful business solutions. Adept at building dashboards, writing complex SQL 
queries, and developing predictive models to solve challenging problems.

Skills:
- Data Visualization: Tableau, Power BI, matplotlib, seaborn
- Statistical Analysis: Python, R, Excel
- Database Management: SQL, MySQL, PostgreSQL
- Machine Learning: Scikit-learn, TensorFlow
- Business Intelligence: Building dashboards, reports, and ad hoc analysis

Work Experience:
Data Analyst | XYZ Corporation | Jan 2020 - Present
- Designed and maintained interactive dashboards for tracking key business metrics.
- Conducted statistical analysis to support marketing strategies, resulting in a 15% increase in ROI.
- Developed machine learning models to predict customer churn, achieving an 85% accuracy rate.
- Collaborated with IT and business teams to streamline data pipelines.

Junior Data Analyst | ABC Tech | Jun 2018 - Dec 2019
- Analyzed customer behavior using SQL and Python to optimize product offerings.
- Automated data extraction processes, reducing report generation time by 30%.
- Created visualizations to communicate findings to stakeholders.

Education:
- Bachelor of Science in Statistics | University of Data Science | 2018

Certifications:
- Tableau Desktop Specialist
- Google Data Analytics Professional Certificate"""


#BASIC DATA #eventually maybe make keyword similarity so can put skills by job and find the job / skills if smiliar liek a giant database / tree IDK getting to complicated
predefined_keywords = {
    "tools": ["Python", "SQL", "Tableau", "Power BI"],
    "skills": ["data visualization", "statistical analysis", "predictive modeling"],
    "certifications": ["Google Data Analytics", "Tableau Certified Professional"],
    "qualifications": ["Bachelor's degree", "Statistics", "Computer Science"]
}

high_relevance_keywords = {
    "python", "sql", "tableau", "power bi", "data visualization",
    "predictive modeling", "machine learning", "statistics", "dashboard",
    "database", "excel", "scikit-learn", "r", "etl", "data pipeline"
}

"""Welcome to the Resume Optimizer!
This program helps you:
1. Parse your resume and a job description.
2. Identify keywords to improve your match score.
3. Generate a tailored LLM (ChatGPT) prompt to rewrite your resume.
4. Save and manage your updated resume automatically.

You can:
- Enter your resume and job description directly.
- Use default testing mode for examples.
- Save results to a .docx file.

Let's get started!"""

print("Settings Loaded\n") 
print(f"Loading Complete: 100%\n\n{'─' * 80}\n\n")

# INPUTS
def extract_text_from_docx(file_path):
    """
    Extracts and returns the text from a Word document.
    """
    
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

#TODO Develop
def extract_text_from_job_description(url):
    """
    Extracts and returns the text from a job posting URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Example: Extract text from a specific HTML tag
        # Modify the tag and class based on the structure of the job website
        job_description_text = soup.find_all(['p', 'div'], class_=None)
        
        # Join all extracted text into a single string
        text = " ".join([element.get_text() for element in job_description_text])
        return text
    else:
        return f"Failed to fetch URL. Status code: {response.status_code}"

# BACKEND / INTERNAL PROCESSES # Why: Seperation of Concerns / Single Responsibility, Maintainability, Testability, Reusability

def extract_keywords(text):
    """
    Extracts keywords (nouns, verbs, and adjectives) from the input text.
    """
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ")]
    return Counter(keywords)

def extract_keywords_table(doc, verbose=False) -> str:
        """
        Extracts keywords and optionally generates a table of extracted keywords.
        
        Args:
        - doc (spacy.tokens.Doc): A SpaCy Doc object.

        Returns:
        - output (str): A formatted string representing the keyword table.
        """
        # Initialize the table with headers
        output = """TESTING - Extracted Keywords Table:\n"""

        # Loop through tokens in the Doc and append rows to the table
        for token in doc:
            output += f"TEXT: {token.text} | pos: {token.pos_}\n"
    
        #print("Filtered Kywords test", list(filter(lambda x: x.pos_ in ("NOUN", "VERB", "ADJ"), doc)))
        return output

    # Assuming `doc` is already a processed SpaCy Doc object
    #print(extract_keywords_table(doc))

def filter_relevant_keywords(keywords, relevance_criteria):
    """
    Filters keywords based on relevance criteria.
    
    Args:
    - keywords (set): The set of keywords to filter.
    - relevance_criteria (list): A list of high-relevance keyword categories (e.g., skills, tools).
    
    Returns:
    - filtered_keywords (set): A set of keywords that match the relevance criteria.
    """
    # Example criteria: Modify as needed
    high_relevance_keywords = {"tools", "technologies", "skills", "data", "analysis", "visualization",
                               "python", "sql", "tableau", "power bi", "predictive", "machine learning"}
    
    # Filter the keywords
    filtered_keywords = {word for word in keywords if word.lower() in high_relevance_keywords}
    return filtered_keywords

#TODO finish and make better
def compare_keywords(job_keywords, resume_keywords):
    """
    Compares job description keywords with resume keywords.
    Returns a match score and missing keywords.
    """
    job_set = set(job_keywords.keys())
    resume_set = set(resume_keywords.keys())
    
    #MODIFY TO CATCH / CONSIDER FOR SYNONMS?? but not semantics cus jsut too broad
    matched_keywords = job_set & resume_set
    missing_keywords = job_set - resume_set
    
    # Filter matched and missing keywords
    relevant_matched = filter_relevant_keywords(matched_keywords, high_relevance_keywords)
    relevant_missing = filter_relevant_keywords(missing_keywords, high_relevance_keywords)

    match_score = len(matched_keywords) / len(job_set) * 100
    return match_score, matched_keywords, missing_keywords

# Empowers users to use tools like ChatGPT, without me having integrating an API which can be costly or have the program run an oncomputer LLM which is unrealistic and time consuming 
def generate_llm_prompt(resume_text, missing_keywords):
    """
    Generates a prompt for a user to optimize their aresume with only relevant keywords using an LLM like ChatGPT.

    Args:
    - resume_text (str): The current text of the user's resume.
    - missing_keywords (set): A set of missing keywords.

    Returns:
    - str: A formatted LLM prompt.
    """
    prompt = f"""
I have the following resume:

{resume_text}

However, it is missing the following important keywords relevant to a job description I am targeting:

{', '.join(missing_keywords)}

Please rewrite my resume to include these keywords in a natural, professional, and concise way while maintaining readability. Ensure the updated resume still accurately reflects my skills and experience.
"""
    return prompt

def output_results(resume_text, job_description_text, match_score, matched_keywords, missing_keywords, resume_doc_filepath=None, job_post_url=None) -> str:
    """outputs results and feedback from analysis"""
    results = f"""
{'─' * 80}


PROGRAM OUTPUT:


{'─' * 80}


ANALYSIS:
Match Score: {match_score:.2f}%


{'─' * 80}


Matched Keywords:
{'\n'.join(['* ' + keyword for keyword in matched_keywords])}
Skills, tools, quali, other relevant terms


{'─' * 80}

Missing Keywords:
{'\n'.join(['* ' + keyword for keyword in missing_keywords])}
Skills, tools, quali, other relevant terms


{'─' * 80}

LLM PROMPT:
Please use the following prompt with your preferred language model (e.g., ChatGPT) to improve your resume:

{generate_llm_prompt(resume_text, missing_keywords)}


{'─' * 80}


Instructions:
Copy the entire prompt and paste it into the language model interface.
Review the generated resume carefully to ensure accuracy and that it truly represents your skills and experience.


{'─' * 80}


Additional Tips:
Authenticity: Only include keywords that genuinely reflect your abilities and experiences.
Specificity: Where possible, provide specific examples or achievements that align with the missing keywords.
Formatting: After updating your resume, ensure that the formatting remains professional and easy to read.


{'─' * 80}
    """
    
    return results

def process_inputs(inputs: dict) -> tuple:
    """
    Process and validate the collected inputs.
    Args:
        inputs (dict): Raw user inputs.
    Returns:
        tuple: (default_testing, job_post_url, resume_doc_filepath) Processed configuration values.
    Raises:
        ValueError: If required inputs are invalid or missing.
    """
    default_testing = inputs.get('default_testing', True)
    job_post_url = inputs.get('job_post_url')
    resume_doc_filepath = inputs.get('resume_doc_filepath')

    if not default_testing:
        # Validate resume file path
        if not resume_doc_filepath:
            raise ValueError("ERROR - Resume file path is required when not in default testing mode.")
        elif not os.path.isfile(resume_doc_filepath):
            raise ValueError(f"ERROR - Resume file '{resume_doc_filepath}' does not exist.")
        
        if not job_post_url:
                    raise ValueError("ERROR - Job post URL is required when not in default testing mode.")
    else:
        resume_doc_filepath = None
        job_post_url = None
        
    return default_testing, job_post_url, resume_doc_filepath

def run_program(
    default_testing=True,
    basic_job_description=None,
    basic_resume_text=None,
    resume_doc_filepath=None,
    job_post_url=None
) -> str:
    """
    Run the ATS program to analyze the match between a resume and a job description.

    Args:
    - default_testing (bool): Whether to run in testing mode with basic inputs.
    - basic_job_description (str): Basic job description text for testing.
    - basic_resume_text (str): Basic resume text for testing.
    - resume_doc_filepath (str): File path to a resume in .docx format.
    - job_post_url (str): URL to a job posting.

    Returns:
    - output (str): A formatted string containing the analysis results.
    """
    
    try:
        # Select Inputs
        if default_testing:
            if not basic_job_description or not basic_resume_text:
                raise ValueError("ERROR - Missing basic testing inputs: job description or resume text.")
            job_description_text = basic_job_description
            resume_text = basic_resume_text
        else:
            # Production mode: Load inputs from files and URLs
            if not resume_doc_filepath:
                raise ValueError("ERROR - Resume file path is required in production mode.")
            if not job_post_url:
                raise ValueError("ERROR - Job post URL is required in production mode.")

            resume_text = extract_text_from_docx(resume_doc_filepath)
            job_description_text = extract_text_from_job_description(job_post_url)

            if not resume_text:
                raise ValueError("ERROR - Unable to extract text from resume file.")
            if not job_description_text:
                raise ValueError("ERROR - Unable to extract text from job posting.")

        # Extract keywords
        job_keywords = extract_keywords(job_description_text)
        resume_keywords = extract_keywords(resume_text)

        # Compare keywords
        match_score, matched_keywords, missing_keywords = compare_keywords(job_keywords, resume_keywords)
 
        output = output_results(
            resume_text,
            job_description_text,
            match_score,
            matched_keywords,
            missing_keywords,
            resume_doc_filepath,
            job_post_url
        )

        return output
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"ERROR - Failed to generate output: {str(e)}"
    
#FRONT END
def get_user_input(prompt: str, default_value: any) -> any:
    user_input = input(f"{prompt}: ")
    return user_input.strip() or default_value

def get_user_inputs(resume_doc_filepath: str = None, job_post_url: str = None, default_testing: bool = None) -> dict:
    """
    Collect user inputs for program configuration.
    Returns:
        dict: A dictionary of raw inputs.
    """
    user_input = get_user_input(
        f"Enter Basic Testing Mode (Y/N)? (Default: {'Y' if default_testing else 'N'})",
        default_testing
        )
    
    inputs = {}
    
    # WHY - If the user gives no input the default value is a boolean (or if any non string data) and you can't use .lower() on a boolean so to prevent an error this makes sure it is only done on a string
    if isinstance(user_input, str):
        negatory = ["n", "no", "f", "false", "False"]
        inputs['default_testing'] = user_input.strip().lower() not in negatory
    else:
        inputs['default_testing'] = user_input
        
    if not inputs['default_testing']:
        inputs['resume_doc_filepath'] = get_user_input(
            f"Enter resume file path (Default: {resume_doc_filepath})", 
            resume_doc_filepath
        )
        
        inputs['job_post_url'] = get_user_input(
            f"Enter job posting URL (Default: {job_post_url})", 
            job_post_url
        )
    else:
        inputs['resume_doc_filepath'] = None
        inputs['job_post_url'] = None

    inputs_list = ""
    inputs_list = "\n".join([f"{entry}: {value}" for entry, value in inputs.items()]) 
    logged_inputs = f"""\n
        --- Inputs Collected ---
        {inputs_list}
        -------------------------
        \n"""
    print(logged_inputs)
    
    return inputs

# EXECUTION BLOCK | INTEGRATION - FRONTEND AND BACKEND
