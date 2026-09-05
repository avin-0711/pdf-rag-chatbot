import os

import streamlit as st


def get_setting(name: str) -> str | None:
    """Read a setting from the environment or Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    return str(value) if value else None
