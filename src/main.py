import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n================ Running {script_path} ================")
    # Determine the python executable (check for virtual environment first)
    if sys.platform == "win32":
        venv_python = os.path.join(".venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(".venv", "bin", "python")
        
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    
    res = subprocess.run([python_exe, script_path])
    if res.returncode != 0:
        raise RuntimeError(f"Script {script_path} failed with exit code {res.returncode}")

def main():
    # Sequence of pipeline execution
    run_script("src/data_prep.py")
    run_script("src/bayes_network.py")
    run_script("src/bnn_model.py")
    run_script("src/evaluate.py")
    print("\n================ Entire pipeline completed successfully! ================")

if __name__ == "__main__":
    main()
