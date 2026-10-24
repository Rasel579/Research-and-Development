import subprocess


def read_file(path):
    with open(path) as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    return f"Записано в {path}"

def run_command(command):
    r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return r.stdout + r.stderr