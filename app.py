import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import traceback
from typing import Dict, List, Any
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Excel Plot Generator", layout="wide")

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found in .env file")
        return None
    return Groq(api_key=api_key)

GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

# Initialize session state
if 'df_dict' not in st.session_state:
    st.session_state.df_dict = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_plot' not in st.session_state:
    st.session_state.current_plot = None

def load_excel_file(uploaded_file) -> Dict[str, pd.DataFrame]:
    """Parse Excel file and return dictionary of DataFrames."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            return {'Sheet1': df}
        elif file_extension in ['xlsx', 'xls']:
            excel_file = pd.ExcelFile(uploaded_file)
            df_dict = {}
            for sheet_name in excel_file.sheet_names:
                df_dict[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
            return df_dict
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

def get_data_summary(df_dict: Dict[str, pd.DataFrame]) -> str:
    """Generate summary of available data for LLM context."""
    summary = []
    for sheet_name, df in df_dict.items():
        summary.append(f"Sheet: {sheet_name}")
        summary.append(f"Columns: {', '.join(df.columns.tolist())}")
        summary.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Add sample data types
        dtypes_str = ", ".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
        summary.append(f"Data types: {dtypes_str}")
        summary.append("")
    return "\n".join(summary)

def generate_matplotlib_code(user_instruction: str, data_summary: str, client: Groq) -> str:
    """Generate Matplotlib code using Groq API."""
    
    if not client:
        raise Exception("Groq client not initialized. Check your API key.")
    
    # Build the prompt for code generation
    prompt = f"""You are a Python matplotlib code generator. Generate complete, executable Python code to create a publication-ready plot based on the user's instruction.

Available Data:
{data_summary}

User Instruction: {user_instruction}

CRITICAL Requirements:
1. Use the variable 'df_dict' which contains DataFrames (keys are sheet names like '{list(st.session_state.df_dict.keys())[0] if st.session_state.df_dict else "Sheet1"}')
   ⚠️ NEVER redefine df_dict! It is already provided with real data. Do NOT write: df_dict = {{...}} or set values to None.
   IMPORTANT - Data Filtering: When filtering data (e.g., x <= 500), ALWAYS create a mask variable FIRST, then apply to both x and y:
   CORRECT:
     mask = df['X'] <= 500
     x = df['X'][mask]
     y = df['Y'][mask]
   WRONG (causes index alignment error):
     x = x[x <= 500]
     y = y[x <= 500]  # ERROR: x is already filtered, indices don't match!
2. Import necessary libraries at the top (matplotlib.pyplot as plt, numpy as np)
3. Access data like: df = df_dict['SheetName'] - the df_dict is already populated with actual DataFrames
4. Create publication-ready plots with EXACT styling:
   - Figure size: figsize=(14, 4) for wide aspect ratio
   - Line plots: use navy blue color ('#1f3b73' or similar)
   - Fill under curve: use light blue fill (ax.fill_between with alpha=0.3-0.4)
   - Grid: ALWAYS include grid with white lines (color='white', alpha=0.7, linewidth=0.8)
   - Background: light grayish-blue (ax.set_facecolor('#E8EBF5') or similar)
   - Spine styling: visible, navy color, linewidth=1.2
   - Font sizes: xlabel/ylabel=11
   - Remove top and right spines: ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
   - X-axis limits: ALWAYS set x-axis to start at 0 and end at the maximum x value (ax.set_xlim(left=0, right=max(x))) to remove blank space
   - DO NOT add any title to the plot (no ax.set_title() or fig.suptitle())
5. Handle smoothing if requested using numpy (moving average or savgol_filter)
6. Store the figure in variable 'fig' and axes in 'ax'
7. Use plt.subplots() to create fig and ax
8. Do NOT call plt.show()
9. Always use tight_layout()
10. NEVER add titles to the plot - no titles should be displayed

STYLE TEMPLATE (use this exact styling):
```
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(x, y, color='#1f3b73', linewidth=1.5)
ax.fill_between(x, y, alpha=0.35, color='#4a7ba7')
ax.set_facecolor('#E8EBF5')
ax.grid(True, color='white', alpha=0.7, linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#1f3b73')
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#1f3b73')
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(colors='#1f3b73', which='both')
ax.set_xlabel('xlabel', fontsize=11, color='#1f3b73')
ax.set_ylabel('ylabel', fontsize=11, color='#1f3b73')
```

RESPONSE FORMAT:
Return ONLY executable Python code with no markdown formatting, no ```python blocks, no explanations.
Start directly with import statements.
"""

    try:
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python programmer specializing in matplotlib visualization. Generate clean, executable code without any markdown formatting or explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=2000
        )
        
        code = chat_completion.choices[0].message.content.strip()
        
        # Clean up code (remove markdown if present)
        if code.startswith('```python'):
            code = code.split('```python')[1]
        if code.startswith('```'):
            code = code.split('```')[1]
        if code.endswith('```'):
            code = code.rsplit('```', 1)[0]
        
        code = code.strip()
        
        # Sanitize code: Remove any lines that try to redefine df_dict
        code = sanitize_generated_code(code)
        
        return code
        
    except Exception as e:
        raise Exception(f"Error calling Groq API: {str(e)}")

def sanitize_generated_code(code: str) -> str:
    """Remove problematic lines from LLM-generated code."""
    lines = code.split('\n')
    sanitized_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip lines that redefine df_dict (e.g., "df_dict = {...}" or "df_dict = {'name': None}")
        if stripped.startswith('df_dict') and '=' in stripped and '{' in stripped:
            continue
        # Skip comment lines about replacing with actual DataFrames
        if '# replace with actual' in stripped.lower():
            continue
        sanitized_lines.append(line)
    
    return '\n'.join(sanitized_lines)

def execute_plot_code(code: str, df_dict: Dict[str, pd.DataFrame]) -> plt.Figure:
    """Safely execute generated matplotlib code and return figure."""
    try:
        # Create execution environment
        exec_globals = {
            'plt': plt,
            'np': np,
            'df_dict': df_dict,
            'pd': pd
        }
        
        # Execute code
        exec(code, exec_globals)
        
        # Retrieve figure
        if 'fig' in exec_globals:
            fig = exec_globals['fig']
            
            # Post-process: Set x-axis limits to remove blank space
            # Start at 0 and end at the maximum x value in the plot
            for ax in fig.axes:
                # Get all line/plot data to find actual x range
                lines = ax.get_lines()
                collections = ax.collections  # For fill_between and similar
                
                all_x_data = []
                
                # Collect x data from lines
                for line in lines:
                    x_data = line.get_xdata()
                    if len(x_data) > 0:
                        all_x_data.extend(x_data)
                
                # Collect x data from collections (fill_between, scatter, etc.)
                for collection in collections:
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        all_x_data.extend(offsets[:, 0])
                
                # Also check patches (bars, rectangles, etc.)
                patches = ax.patches
                for patch in patches:
                    x_pos = patch.get_x()
                    width = patch.get_width()
                    all_x_data.extend([x_pos, x_pos + width])
                
                if all_x_data:
                    max_x = max(all_x_data)
                    # Always start at 0 and end at max x value
                    ax.set_xlim(left=0, right=max_x)
                
                # Set axis labels
                ax.set_xlabel('mm', fontsize=11, color='#1f3b73')
                ax.set_ylabel('cd/m²', fontsize=11, color='#1f3b73')
                
                # Remove any titles
                ax.set_title('')
            
            # Remove figure-level titles (suptitle)
            fig.suptitle('')
            
            return fig
        else:
            raise Exception("Generated code did not create 'fig' variable")
    
    except Exception as e:
        raise Exception(f"Error executing plot code: {str(e)}\n\nGenerated code:\n{code}")

def fig_to_bytes(fig: plt.Figure, dpi: int = 100) -> bytes:
    """Convert matplotlib figure to high-resolution PNG bytes (1400x400px at 100dpi)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='#E8EBF5')
    buf.seek(0)
    return buf.getvalue()

# UI Layout
st.title("📊 Excel Plot Generator")
st.markdown("Upload an Excel file and describe your desired plot in natural language.")

# Initialize Groq client
groq_client = get_groq_client()

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Data")
    
    # Display API status
    if groq_client:
        st.success(f"✅ Groq API Connected")
        st.caption(f"Model: {GROQ_MODEL}")
    else:
        st.error("❌ Groq API Not Configured")
        st.caption("Add GROQ_API_KEY to .env file")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Choose Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your data file to get started"
    )
    
    if uploaded_file:
        try:
            df_dict = load_excel_file(uploaded_file)
            st.session_state.df_dict = df_dict
            st.success(f"✅ Loaded {len(df_dict)} sheet(s)")
            
            # Display data preview
            st.subheader("Data Preview")
            for sheet_name, df in df_dict.items():
                with st.expander(f"📄 {sheet_name}"):
                    st.dataframe(df.head(), width='stretch')
                    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ {str(e)}")
            st.session_state.df_dict = {}

# Main area
if st.session_state.df_dict and groq_client:
    # Chat input
    user_input = st.chat_input("Describe the plot you want to create...")
    
    if user_input:
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        try:
            # Generate data summary
            data_summary = get_data_summary(st.session_state.df_dict)
            
            # Generate matplotlib code using Groq
            with st.spinner("Generating plot code with Groq AI..."):
                plot_code = generate_matplotlib_code(user_input, data_summary, groq_client)
            
            # Execute code and create plot
            with st.spinner("Creating plot..."):
                fig = execute_plot_code(plot_code, st.session_state.df_dict)
                st.session_state.current_plot = fig
            
            # Add success to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "✅ Plot created successfully!",
                "code": plot_code
            })
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg
            })
    
    # Display chat history
    st.subheader("💬 Chat History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "code" in message:
                with st.expander("View generated code"):
                    st.code(message["code"], language="python")
    
    # Display current plot
    if st.session_state.current_plot:
        st.subheader("📈 Generated Plot")
        st.pyplot(st.session_state.current_plot)
        
        # Download button
        col1, col2 = st.columns([1, 4])
        with col1:
            img_bytes = fig_to_bytes(st.session_state.current_plot, dpi=100)
            st.download_button(
                label="⬇️ Download PNG",
                data=img_bytes,
                file_name="plot.png",
                mime="image/png"
            )
        
        # Clear plot button
        with col2:
            if st.button("🗑️ Clear Plot"):
                st.session_state.current_plot = None
                st.rerun()

elif not groq_client:
    st.warning("⚠️ Please configure Groq API in .env file")
    st.code("""# Create a .env file with:
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile""")
else:
    st.info("👈 Please upload an Excel or CSV file to get started.")
    
    # Example instructions
    with st.expander("📖 Example Instructions"):
        st.markdown("""
        - "Create a line plot with Temperature on x-axis and Pressure on y-axis"
        - "Plot Sales vs Time with smoothing window of 10"
        - "Make a scatter plot of Height vs Weight with grid"
        - "Bar chart showing Revenue by Quarter with legend"
        - "Plot column A against column B with smooth curve and title 'My Analysis'"
        - "Show histogram of Age distribution"
        """)

# Footer
st.markdown("---")
st.caption("Built for Dcontour Litetech Pvt. Ltd.")
