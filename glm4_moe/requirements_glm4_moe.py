from packaging import version
from transformers import __version__ as transformers_version


def is_accepted():
    return version.parse(transformers_version) >= version.parse('4.55.0')
