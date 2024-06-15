import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Tiêu đề ứng dụng
st.title('Emotion Analysis')

# Import file từ hệ thống
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Đọc dữ liệu từ tập tin CSV hoặc Excel
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error('Only CSV and Excel files are supported.')
            st.stop()
            
        # Loại bỏ cột "Unnamed" nếu có
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        required_columns = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        if not all(col in df.columns for col in required_columns):
            st.error('The file must contain the following columns: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.')
            st.stop()

        #Total Emotions
        st.subheader('Total Emotions')
        fig, ax = plt.subplots()
        df.sum().plot(kind='bar', color=sns.color_palette('pastel', len(df.columns)), ax=ax)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Total')
        ax.set_xticklabels(df.columns, rotation=45)
        
        handles = [Patch(color=color, label=emotion) for emotion, color in zip(df.columns, sns.color_palette('pastel', len(df.columns)))]
        ax.legend(handles=handles, title='Emotion', title_fontsize='small', fontsize='small', loc='upper right')

        st.pyplot(fig)

        #Average Emotions
        st.subheader('Average Emotions')
        fig, ax = plt.subplots()
        df.mean().plot(kind='bar', color=sns.color_palette('pastel', len(df.columns)), ax=ax)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Average')
        ax.set_xticklabels(df.columns, rotation=45)
        
        handles = [Patch(color=color, label=emotion) for emotion, color in zip(df.columns, sns.color_palette('pastel', len(df.columns)))]
        ax.legend(handles=handles, title='Emotion', title_fontsize='small', fontsize='small', loc='upper right')

        st.pyplot(fig)

        #Emotion Distribution
        st.subheader('Emotion Distribution')
        fig, ax = plt.subplots(facecolor='none')
        emotions_sum = df.sum()
        percentages = [f'{count/emotions_sum.sum()*100:.1f}%' for count in emotions_sum]
        labels = [f'{emotion} ({percentage})' for emotion, percentage in zip(df.columns, percentages)]
        ax.pie(emotions_sum, colors=sns.color_palette('pastel', len(df.columns)), autopct='', startangle=90)
        ax.axis('equal')

        handles = [Patch(color=color, label=label) for label, color in zip(labels, sns.color_palette('pastel', len(df.columns)))]
        ax.legend(handles=handles, title_fontsize='small', fontsize='small', bbox_to_anchor=(1, 0.5), loc='center left')

        st.pyplot(fig)
                

        #DATASET
        st.subheader('Dataset')
        st.write(df)

    except Exception as e:
        st.error(f'Error: {e}')
