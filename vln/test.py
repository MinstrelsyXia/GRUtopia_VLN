import os
def isDocker():
    return os.path.exists('/.dockerenv')

print(isDocker())