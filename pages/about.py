import streamlit as st
import matplotlib.pyplot as plt
from config.config import NAME

st.set_page_config(page_title="About",
                    page_icon="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3QgeD0iNyIgeT0iMiIgd2lkdGg9IjIiIGhlaWdodD0iMTIiIGZpbGw9ImJsdWUiLz48cmVjdCB4PSIyIiB5PSI3IiB3aWR0aD0iMTIiIGhlaWdodD0iMiIgZmlsbD0iYmx1ZSIvPjwvc3ZnPg==",
                   layout="wide")

st.sidebar.title('Training')

st.markdown('# Training\n In this section you can see how the model was trained and the results\n## Dev set Confusion matrix on the model\n - In simple terms the dev set is a sample of data that is used to test on and choose between models to compare and contrast which model is best.\n - A confusion matrix just shows which categories the model confused about')


fig = plt.figure(figsize=(4,4))
img = plt.imread(f'training/training_data/confusion_matrix_{NAME}.png')
plt.imshow(img)

st.pyplot(fig, use_container_width=False, )


footer="""<style>
a:link , a:visited{
color: #aaaaff;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: navy-blue;
color: #ccc;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/yehan-dineth/" target="_blank">Yehan Dineth</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)