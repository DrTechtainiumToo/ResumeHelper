__author__ = "Andrew Dybala"
__copyright__ = "Copyright Restricted"
__credits__ = ["Andrew Dybala", "GPT4 as assistant"]
__license__ = "License Name and Info"
__version__ = "1.0.1"
__maintainer__ = "Andrew Dybala"
__email__ = "andrew@dybala.com"
__status__ = "In development"
__compiler__ = "Python 3.12.0"

if __name__ == "__main__":
    
    #Configure settings / collect user inputs
    raw_inputs = get_user_inputs(resume_doc_filepath, job_post_url, default_testing)
        
    try:
        default_testing, job_post_url, resume_doc_filepath = process_inputs(raw_inputs)
        output = run_program(
            default_testing=default_testing,
            basic_job_description=basic_job_description,
            basic_resume_text=basic_resume_text,
            job_post_url=job_post_url,
            resume_doc_filepath=resume_doc_filepath
        )
        print(output)
    except ValueError as e:
        print(str(e))