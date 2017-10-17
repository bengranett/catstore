from metafil import GitEnv

__version__ = '?'

try:
    # if we are running in a git repo, look up the hash
    __version__ = GitEnv(__file__).describe()
except:
    # otherwise check for a version file
    try:
        from . version import __version__
    except:
        pass
