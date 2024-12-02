import time

@contextmanager #NOTE allow to time ANY block of code, not just functions. I was learning about ocntext managers, still need to learn more.
def time_block(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        message = f"{label}: {end - start} seconds"
        with open("devious_fuckin_project_root/logs/program_times_log.txt", "a") as log_file:  # "a" opens the file in append mode
            log_file.write(message + "\n")
            
def log_to_file(message: str, file_path: str = "devious_fuckin_project_root/logs/program_times_log.txt"):
    with open(file_path, "a") as log_file:  # "a" opens the file in append mode
            log_file.write(message + "\n")