import subprocess

def shutdown_pc():
    # Specify the time in seconds for shutdown
    time_to_shutdown = 20
    
    # Convert the time to an integer value
    time_to_shutdown_int = int(time_to_shutdown)
    
    # Create a command that shuts down the PC after the specified time
    cmd = "shutdown /s /t {}".format(str(time_to_shutdown_int))
    
    # Execute the command using subprocess
    subprocess.run(cmd, shell=True)

# Call the function to shut down the PC
shutdown_pc()