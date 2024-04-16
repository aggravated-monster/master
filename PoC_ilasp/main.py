import os
import subprocess
import psutil

ILASP_BINARY_NAME = "ILASP"
CLINGO_BINARY_NAME = "clingo"
ILASP_OPERATION_SEARCH_SPACE = "search_space"
binary_folder_name = "./src/bin/"
TIMEOUT_ERROR_CODE = 124


def extract_result():
    result = []
    with open('result.tmp') as infile:
        content = infile.readlines()
        for info in content:
            if info.startswith('Pre-processing'):
                break
            result.append(info.strip('\n'))
    return result

def learn():
    ilasp_path = os.path.join(binary_folder_name, ILASP_BINARY_NAME)

    print("Laarning...")

    examples = "ou.las"

    #    os.system(clingopath+" "+files+" --timelimit=300 > result.tmp")
    execution_string = ilasp_path + "  --version=4  "  + examples + " > result.tmp"
    ilasp_process = subprocess.Popen(execution_string, shell=True)
    p = psutil.Process(ilasp_process.pid)
    try:
        p.wait(timeout=600)
    except psutil.TimeoutExpired:
        p.kill()
        print("Learner timeout. Process killed.")
        return None

    result = extract_result()

    return result

if __name__ == "__main__":
    # in this PoC,

    #result = learn_to_jump()
    result = learn()

    for line in result:
        print(line)
