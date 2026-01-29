# core/state.py
def state_key(module_name: str, suffix: str) -> str:
    return f"{module_name.replace(' ', '_').lower()}__{suffix}"