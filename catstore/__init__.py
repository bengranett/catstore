import subprocess

__version__ = ''


try:
    # if we are running in a git repo, look up the hash
    __version__ = subprocess.Popen(
        ('git','--git-dir',os.path.dirname(__file__),'describe','--always'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
    assert __version__
except:
    # otherwise check for a version file
    try:
        from . version import version as __version__
    except:
        pass