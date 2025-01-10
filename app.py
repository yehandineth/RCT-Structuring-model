import streamlit as st
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).parent))

from classify.classify import classify
from processing.preprocessing import Abstract

from processing.data_handling import text_to_dataframe

@st.cache_data
def classify_cache(text):
    return classify(Abstract(text))

@st.cache_data
def web_out(text):
    classified = classify_cache(text)
    html_output = "<div id='copy-box' style='background-color: white; padding: 10px; border: 1px solid #ccc; border-radius: 10px; color: #404040;'>"
    for cls, texts in classified.items():
        html_output += f"<p><strong>{cls}</strong></p>"
        html_output += f"<p>{'<br>'.join(texts)}</p>"
    html_output += "<button style='position: absolute; bottom: 0px; right: 10px; background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 4px; cursor: pointer;' onclick='copyText()'>Copy</button>"
    html_output += "</div>"

    copy_script = """
    <script>
    function copyText() {
        var textToCopy = document.getElementById("copy-box").innerText;
        navigator.clipboard.writeText(textToCopy);
    }
    </script>
    """
    st.markdown(html_output+copy_script, unsafe_allow_html=True)

st.title('PubMed RCT abstract Structuring')

st.sidebar.title('Navigation')

text = st.text_input(
    'Enter your abstract here',
    )

web_out(text)
