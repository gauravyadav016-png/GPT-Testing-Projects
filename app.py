import streamlit as st
import pandas as pd
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Set up Google API Key
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
client = None

if not google_api_key:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
else:
    genai.configure(api_key=google_api_key)
    client = genai.GenerativeModel('gemini-1.5-flash')

def compare_answers(question, correct_answer, llm_answer):
    """
    Use Google's Gemini API to compare the correct answer with the LLM's answer in the context of the question.
    """
    retries = 3
    for i in range(retries):
        try:
            prompt_text = f"Question: {question}\nCorrect Answer: {correct_answer}\nLLM Answer: {llm_answer}\nDoes the LLM answer convey the same meaning as the correct answer? (yes or no)"
            response = client.generate_content(prompt_text)
            # print(response)
            result = response.text.strip().lower()
            return 1 if 'yes' in result else 0
        except Exception as e:
            if "429" in str(e) and i < retries - 1:
                time.sleep(2 ** i)  # Exponential backoff
            else:
                st.error(f"Error while calling Gemini API: {e}")
                return 0
   
def process_file(file):
    # Read the uploaded file into a DataFrame
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    
    # Apply the compare_answers function to each row and set the correction_flag
    df['correction_flag'] = df.apply(
        lambda row: compare_answers(row['questions'], row['correct_answers'], row['answers_by_llm']),
        axis=1
    )
    
    return df

def main():
    st.title('LLM Answer Correction Checker')

    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        st.write("File Uploaded Successfully!")
        
        if st.button('Submit'):
            processed_df = process_file(uploaded_file)
            
            st.write("Processed Data")
            st.dataframe(processed_df)
            
            csv = processed_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download processed data as CSV",
                data=csv,
                file_name='processed_data.csv',
                mime='text/csv',
            )

if __name__ == '__main__':
    main()