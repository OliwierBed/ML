from pathlib import Path
from omegaconf import OmegaConf

def load_config(path: str | None = None):
    """
    Ładuje config/config.yaml. Jeśli path jest None, szuka pliku config.yaml
    obok tego modułu.
    """
    if path is None:
        path = Path(__file__).with_name("config.yaml")
    return OmegaConf.load(path)
