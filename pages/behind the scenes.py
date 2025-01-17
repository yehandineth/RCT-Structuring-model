import streamlit as st
import matplotlib.pyplot as plt
from config.config import NAME, DF
import pandas as pd


st.set_page_config(page_title="Behind The Scenes",
                    page_icon="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3QgeD0iNyIgeT0iMiIgd2lkdGg9IjIiIGhlaWdodD0iMTIiIGZpbGw9ImJsdWUiLz48cmVjdCB4PSIyIiB5PSI3IiB3aWR0aD0iMTIiIGhlaWdodD0iMiIgZmlsbD0iYmx1ZSIvPjwvc3ZnPg==",
                   layout="wide")

st.sidebar.markdown("# **Contents**\n#### Training\n#### Development\n#### Testing\n#### Model Architecture")

fig1 = plt.figure(figsize=(4,4))
img = plt.imread(f'serialization/transition_matrix_{DF}.jpg')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

fig2 = plt.figure(figsize=(4,4))
img = plt.imread(f'training/training_data/confusion_matrix_{NAME}.png')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

fig3 = plt.figure(figsize=(4,4))
img = plt.imread(f'testing/confusion_matrix_{NAME}.png')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

fig4 = plt.figure(figsize=(7,14))
img = plt.imread(f'serialization/{NAME}.png')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

st.markdown('# Training\n---\n>Training data is the subset of data that the model is basically trained on. The below section shows the results and how the model performed and learned from the training data.\n')

st.markdown('## Train set model Performance\n - The model performed pretty well on the training data and this is usually the case in almost any model.\n - This is due to the model learning to fit this dataset very well you can see it has achieved an excellent accuracy On this dataset.\n - But this accuracy cannot be relied on because it is almost like *Writing an exam paper That you have already practiced on before*.')
st.write(pd.read_csv(f'training/training_data/classification_report_{NAME}_train.csv', index_col=0))
st.markdown('## Transition matrix Learned by the model during training\n - The transition matrix is just a table that shows the probability that in a abstract paragraph that a sentence belonging to a particular class showing up next to another class.\n - This is visualized because this is relatively easy to visualize and understand than other parts of the model.')
st.pyplot(fig1, use_container_width=False, )

st.markdown('# Development\n---\n>Development data set is the subset of data that is used to test every model on to choose which is best.\n - This dataset is not used to train the model so in this case this is like a *midterm examination for the model*.')

st.markdown('## Development set model Performance\n - The model relatively well on the development data with an accuray close to the training accuracy.\n - This indicates that this model might generalize well on the required environments.')
st.write(pd.read_csv(f'training/training_data/classification_report_{NAME}_dev.csv', index_col=0))
st.markdown('## Dev set Confusion matrix on the model\n - In simple terms the dev set is a sample of data that is used to test on and choose between models to compare and contrast which model is best.\n - A confusion matrix just shows which categories the model confused about')
st.pyplot(fig2, use_container_width=False, )

st.markdown('# Testing\n---\n> The testing data set is like the *Final Examination* for the model to determine whether it is suitable to use and/or release to the target usage.')

st.markdown('## Testing set model Performance\n - If the model perform almost close to the dev set during testing. It is an indication that the model might generalize well.')
st.write(pd.read_csv(f'testing/classification_report_{NAME}.csv', index_col=0))
st.markdown('## Testing set Confusion matrix on the model')
st.pyplot(fig3, use_container_width=False, )

st.markdown('# The Model Architecture\n---\n>')
st.markdown('> - If anyone is interested the tensorflow model architecture is as follows. The complete pipeline is a little more complicated.')
st.pyplot(fig4, use_container_width=False, )

st.markdown('> You can help make this project better through this [github repo](https://github.com/yehandineth/RCT-Structuring-model)')

st.markdown('# Thank You for using this app❤\n---\n>')

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