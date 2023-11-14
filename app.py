import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util


@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('stsb-roberta-large')

model=load_model()

st.title('Task For Precily')


text1 = st.text_input('Enter  text 1 to check for similarity')
text2 = st.text_input('Enter text 2 to check for similarity')

submit = st.button('Show Similarity')



if submit:

    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

    similarity_score = cosine_score.item()

    st.title(f'Similarity Score:{similarity_score}')

st.header('OR')


file_uploaded = st.file_uploader('Choose a CSV file..', type=['csv'])
submission = st.button('Show Dataframe With Score')
if submission:
    if file_uploaded is not None:
        df = pd.read_csv(file_uploaded)
        df.drop_duplicates(inplace=True)
        t1 = df['text1'].to_numpy()
        t2 = df['text2'].to_numpy()
        semantic_similarity = []
        for i in range(len(df)):
            embed1 = model.encode(t1[i], convert_to_tensor=True)
            embed2 = model.encode(t2[i], convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(embed1, embed2)
            similarity_scores = cosine_scores.item()
            semantic_similarity.append(similarity_scores)
        df['SimilarityScore'] = semantic_similarity
        st.dataframe(df)

        if df[df['SimilarityScore'] > 0.6] is not None:

            st.header('Texts With Very High Similarity Score')

            new_df = df[df['SimilarityScore'] > 0.6]
            st.dataframe(new_df)
