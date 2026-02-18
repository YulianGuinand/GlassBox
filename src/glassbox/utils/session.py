import streamlit as st
from typing import Any, Optional

class SessionManager:
    """Helper to manage st.session_state with type safety and default values."""
    
    @staticmethod
    def init_state(key: str, default_value: Any):
        if key not in st.session_state:
            st.session_state[key] = default_value

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        st.session_state[key] = value

    # --- Specific Accessors for GlassBox ---
    @staticmethod
    def get_data_manager():
        return st.session_state.get("data_manager")

    @staticmethod
    def set_data_manager(manager):
        st.session_state["data_manager"] = manager

    @staticmethod
    def get_trainer():
        return st.session_state.get("trainer")

    @staticmethod
    def set_trainer(trainer):
        st.session_state["trainer"] = trainer

    @staticmethod
    def get_training_queue():
        return st.session_state.get("training_queue")
