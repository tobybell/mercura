import subprocess

interactive = subprocess.Popen(('../c/interactive'), stdout=subprocess.PIPE)
output = subprocess.check_output(('grep', 'process_name'), stdin=interactive.stdout)
interactive.wait()