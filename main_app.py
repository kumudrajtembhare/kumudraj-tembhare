import os
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv
import pyodbc
import time
import asyncio
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt  
from matplotlib.ticker import FuncFormatter                             
import json
from PIL import Image
import base64

def get_base64_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
   
# Load the logo image and convert to base64
logo_path = "logo.jpeg"
logo_base64 = get_base64_image(logo_path)

#Adding the Power BI report link
#bookings_url = "https://app.powerbi.com/groups/me/apps/80507639-1b1f-422e-9a4c-b31cdacd7e2c/reports/b3820e59-92b2-4497-bbb9-4283461ac95e/ReportSection?experience=power-bi"
#Bookings_link_text = "link"
#Bookings_markdown_link = f'[{Bookings_link_text}]({bookings_url})'
#Bookings_pre_text = "For more info. Please click on the Bookings Power BI link: "
 
#PP_url = "https://app.powerbi.com/groups/me/apps/1c5541d2-7d89-4b2a-8f4b-8eed31ee029d/reports/97f83734-2eb3-4611-9ca1-14a2079321bb/ReportSection?experience=power-bi"
#PP_link_text = "link"
#PP_markdown_link = f'[{PP_link_text}]({PP_url})'
#PP_pre_text = "For more info. Please click on the Purchase Product Power BI link: "

links = {
    "https://app.powerbi.com/groups/me/apps/80507639-1b1f-422e-9a4c-b31cdacd7e2c/reports/b3820e59-92b2-4497-bbb9-4283461ac95e/ReportSection?experience=power-bi": ["booking", "acv"],
    "https://app.powerbi.com/groups/me/apps/1c5541d2-7d89-4b2a-8f4b-8eed31ee029d/reports/97f83734-2eb3-4611-9ca1-14a2079321bb/ReportSection?experience=power-bi": ["purchase", "transaction", "tier"]
}

ShareP_links = {
    "https://microsoftapc.sharepoint.com/:f:/s/AI_chatbot_Sharepoint/EgHaUMaCDstInOycfw6KdtcBxosFZX0Az5NLhZYobhZ7OA?e=QXduSZ": ["booking", "acv"],
    "https://microsoftapc.sharepoint.com/:f:/s/AI_chatbot_Sharepoint/EsOM2_zI6IBIkCufnMFg9BkBT0twamnAmTtltaZPO6bDag?e=BMkhPQ":["purchase", "transaction", "tier"]
}
button_css = """
<style>
.button {
    background-color: #f1f1f1; /* Light Gray */
    color: #0078d4; /* Bing Blue */
    border: 1px solid #d1d1d1; /* Light Gray Border */
    border-radius: 4px; /* Rounded Corners */
    padding: 5px 10px; /* Smaller Padding */
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 12px; /* Smaller Font Size */
    cursor: pointer;
    margin: 2px 1px; /* Smaller Margin */
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1); /* Subtle Shadow */
    transition: background-color 0.3s, box-shadow 0.3s; /* Smooth Transition */
}

.button:hover {
    background-color: #e1e1e1; /* Slightly Darker Gray on Hover */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Slightly Larger Shadow on Hover */
}
</style>
"""

logo = Image.open("logo.jpeg")
 
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    .centered img {
        width: 200px;  /* Adjust the width as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
 
st.markdown(
    f"""
    <div class="centered">
        <img src="data:image/png;base64,{logo_base64}" alt="Company Logo">
        <h1>Welcome to Nuance FinPlat Copilot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

def get_link_from_prompt(prompt):
    for link, keywords in links.items():
        for keyword in keywords:
            if keyword in prompt.lower():
                return link
    return None


def get_SP_link_from_prompt(prompt):
    for link, keywords in ShareP_links.items():
        for keyword in keywords:
            if keyword in prompt.lower():
                return link
    return None

def analyze_prompt_for_chart(prompt):
    chart_types = ['bar chart', 'pie chart']
    for chart_type in chart_types:
        if chart_type in prompt.lower():
            return chart_type
    return None

def render_bar_chart(df):
    # Convert columns to numeric where possible
    #df = df.apply(pd.to_numeric, errors='ignore')
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue
    
    # Select numeric and object (string) columns
    numeric_columns = df.select_dtypes(include='number').columns
    string_columns = df.select_dtypes(include='object').columns

    if len(numeric_columns) < 1 or len(string_columns) < 1:
        st.write("Not enough data to plot a bar chart.")
    elif len(numeric_columns) == 1 and len(string_columns) >= 1:    
        fig, ax = plt.subplots()
        df.plot(kind='bar', x=string_columns[0], y=numeric_columns[0], ax=ax)
        ax.set_title('Bar Chart')
        ax.set_xlabel(string_columns[0])
        ax.set_ylabel(numeric_columns[0])
        # Format y-axis to show natural numbers
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        df.plot(kind='bar', x=numeric_columns[0], y=numeric_columns[1], ax=ax)
        ax.set_title('Bar Chart')
        ax.set_xlabel(numeric_columns[0])
        ax.set_ylabel(numeric_columns[1])
        # Format y-axis to show natural numbers
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        st.pyplot(fig)
        #st.bar_chart(df[numeric_columns])

def render_pie_chart(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue
    numeric_columns = df.select_dtypes(include='number').columns
    if len(numeric_columns) < 1:
        st.write("Not enough numeric data to plot a pie chart.")
    else:
        y_column = numeric_columns[0]
        if df.columns[0] == y_column:
            y_column = numeric_columns[1] if len(numeric_columns) > 1 else df.columns[1]
        fig, ax = plt.subplots()
        df.set_index(df.columns[0]).plot(kind='pie', y=y_column, autopct='%1.1f%%', ax=ax)
        ax.set_title('Pie Chart')
        st.pyplot(fig)  

def refine_prompt_input(input):
    # Remove chart type keywords to maintain consistent query generation
    input = input.lower().replace('bar chart', '').replace('pie chart', '').strip()
    return input       

def semanticFunctions(kernel, skills_directory, skill_name, input):
    functions = kernel.import_semantic_skill_from_directory(skills_directory, "plugins")
    summarizeFunction = functions[skill_name]
    if summarizeFunction:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Execute the skill function with the provided input_data
            # # Ensure consistent query generation by refining prompt inputs
            refined_input = refine_prompt_input(input)
            result = summarizeFunction(input)
        finally:
            # Close the event loop
            loop.close()
        
        return result
    else:
        return f"Skill '{skill_name}' not found in the specified directory."
        
# Function to get the result from the database
def get_result_from_database(sql_query):
    server_name = os.getenv('SERVER_NAME')
    database_name = os.getenv('DATABASE_NAME')
    username = os.getenv('SQLADMIN_USER')
    password = os.getenv('SQL_PASSWORD')
    conn = pyodbc.connect('DRIVER={driver};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}'.format(driver="ODBC Driver 18 for SQL Server",server_name=server_name, database_name=database_name, username=username, password=password))
    
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in result]
        df = pd.DataFrame(data)
    except:
        df = pd.DataFrame()  # Return an empty DataFrame if no results are found
    cursor.close()
    conn.close()
    return df

def main():

    #Load environment variables from .env file
    load_dotenv()
    #st.set_option("server.port", 8000)

    # Create a new kernel
    kernel = sk.Kernel()
    context = kernel.create_new_context()
    context['result'] = ""

    # Configure AI service used by the kernel
    #deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

    # Debugging: Print the values of the environment variables
    st.write(f"Deployment: {deployment}")
    st.write(f"API Key: {api_key}")
    st.write(f"Endpoint: {endpoint}")

    # Check if any of these are None
    if not deployment or not api_key or not endpoint:
        st.error("One or more environment variables are not set. Please check your configuration.")
        return

    # Add the AI service to the kernel
    kernel.add_text_completion_service("dv", AzureChatCompletion(deployment_name=deployment, endpoint=endpoint, api_key=api_key))
                                   
    #create storage message
    if "messages"   not in  st.session_state:
        st.session_state.messages=[]
    
    #display  chat history
    #st.write(st.session_state.messages)
    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            if isinstance(message.get("content"), pd.DataFrame):
                st.write(message.get("content").to_html(index=False), unsafe_allow_html=True)
            else:
                st.write(message.get("content"))
    prompt = st.chat_input("Hello, How i can help you")
    if prompt:
            #append the messages
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                query=st.write(prompt)
            # Build the conversation history
            #conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-3:]])
        
            # Pass the entire conversation history to the semantic function
            skills_directory = "."
            #sql_query = semanticFunctions(kernel, skills_directory, "nlpToSQLPlugin", conversation_history)
            sql_query = semanticFunctions(kernel, skills_directory, "nlpToSQLPlugin", prompt)
            #full_sql_query = sql_query.result                                     
            full_sql_query = sql_query.result.split(';')[0]
                                            
            # Remove the note if it exists
            if 'Note:' in full_sql_query:
                full_sql_query = full_sql_query.split('(Note:')[0].strip()                                             

            st.session_state.messages.append({"role":"assistant","content":full_sql_query})
            with st.chat_message("assistant"):
                                     
                #st.text_area("SQL Query", value=full_sql_query)
                st.code(full_sql_query, language='sql')  # Display SQL query with formatting
       
        # Use the query to call the database and get the output
            result = get_result_from_database(full_sql_query)
                                          
            st.session_state.messages.append({"role":"answer","content":result})
            with st.chat_message("answer"):
                    if result.empty:
                        st.write("No Result Found")
                    else:
                        pd.options.display.float_format = '{:.2f}'.format  # Format floats with 2 decimal places
                        st.write(result.reset_index(drop=True).to_html(index=False), unsafe_allow_html=True)
                        chart_type = analyze_prompt_for_chart(prompt)
                        if chart_type and 'bar chart' in chart_type.lower():
                            if 'Bar Chart' in full_sql_query:
                                st.write("Generating a bar chart requires additional data processing tools.")
                            else:
                                render_bar_chart(result)
                        elif chart_type and 'pie chart' in chart_type.lower():
                            if 'Pie Chart' in full_sql_query:
                               st.write("Generating a pie chart requires additional data processing tools.")
                            else:
                               render_pie_chart(result)
                        #if "booking" in prompt.lower():
                        #    st.write(Bookings_pre_text,Bookings_markdown_link)
                        #else:
                        #    st.write(PP_pre_text,PP_markdown_link) 
                        link = get_link_from_prompt(prompt)
                        ShareP_links = get_SP_link_from_prompt(prompt)

                        PowerBI_markdown_link = f'<a href="{link}" target="_blank" class="button">1. 📃PowerBI Link</a>'
                        SharePoint_markdown_link = f'<a href="{ShareP_links}" target="_blank" class="button">2. 📃SharePoint Link</a>'
                        
                        st.markdown(button_css, unsafe_allow_html=True)
                        st.write("For more info, please click on the" + " " + PowerBI_markdown_link + " " + SharePoint_markdown_link, unsafe_allow_html=True)

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time taken Overall(mins): ", (time.time() - start)/60)
