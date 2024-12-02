#maybe put some of these in data



#Default Settings
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

#program setup intro text
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