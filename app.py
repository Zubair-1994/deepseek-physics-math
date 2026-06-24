import os
import json
import re
from datetime import datetime
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------
# CONSTANTS & RUNTIME INITIALIZATION
# ---------------------------------------------------------
HISTORY_DIR = "chat_histories"
TITLES_FILE = os.path.join(HISTORY_DIR, "chat_titles.json")
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# Page Layout configuration
st.set_page_config(page_title="DeepSeek LaTeX Chatbot", layout="wide")
st.title("🤖 DeepSeek Physics & Math")

def get_client():
    user_api_key = st.sidebar.text_input("DeepSeek API Key", type="password")

    try:
        owner_api_key = st.secrets.get("DEEPSEEK_API_KEY")
    except FileNotFoundError:
        owner_api_key = None

    api_key = user_api_key or owner_api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("Enter a DeepSeek API key in the sidebar to use the app.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

client = get_client()

DISPLAY_MATH_PATTERN = re.compile(r"(?s)(\$\$(.*?)\$\$|\\\[(.*?)\\\])")

def normalize_latex_text(text: str) -> str:
    """
    Normalizes common inline math delimiters while leaving display math blocks
    available for st.latex rendering.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    return text

def clean_display_latex(latex: str) -> str:
    latex = latex.strip()
    latex = re.sub(r"\\begin\{align\*?\}", r"\\begin{aligned}", latex)
    latex = re.sub(r"\\end\{align\*?\}", r"\\end{aligned}", latex)

    for env in ("equation", "equation*", "displaymath"):
        begin = rf"\begin{{{env}}}"
        end = rf"\end{{{env}}}"
        if latex.startswith(begin) and latex.endswith(end):
            return latex[len(begin):-len(end)].strip()
    return latex

def render_response(text: str):
    """
    Renders markdown text and display equations separately. This is more stable
    on Streamlit Cloud than sending mixed markdown and $$...$$ blocks through
    st.markdown in one pass.
    """
    text = normalize_latex_text(text)
    cursor = 0

    for match in DISPLAY_MATH_PATTERN.finditer(text):
        markdown_chunk = text[cursor:match.start()]
        if markdown_chunk.strip():
            st.markdown(markdown_chunk)

        latex = match.group(2) if match.group(2) is not None else match.group(3)
        st.latex(clean_display_latex(latex))
        cursor = match.end()

    trailing_chunk = text[cursor:]
    if trailing_chunk.strip():
        st.markdown(trailing_chunk)

# ---------------------------------------------------------
# CONVERSATION STORAGE HANDLERS
# ---------------------------------------------------------
def save_chat_to_disk(session_id, messages):
    if len(messages) > 1:  # Don't persist empty systemic states
        file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)

def list_saved_sessions():
    files = [
        f.replace(".json", "")
        for f in os.listdir(HISTORY_DIR)
        if f.endswith(".json") and f != os.path.basename(TITLES_FILE)
    ]
    return sorted(files, reverse=True)

def load_chat_titles():
    if not os.path.exists(TITLES_FILE):
        return {}
    try:
        with open(TITLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def save_chat_titles(titles):
    with open(TITLES_FILE, "w", encoding="utf-8") as f:
        json.dump(titles, f, indent=4, ensure_ascii=False)

def default_chat_title(session_id):
    return session_id.replace("chat_", "Chat ") if session_id.startswith("chat_") else session_id

SYSTEM_PROMPT = """You are a careful mathematics and physics tutor for undergraduate and graduate students.

Your main job is to derive equations, solve exercises, and explain reasoning step by step. Adapt the depth and pace to the user's level and the problem's difficulty.

Core behavior:
- Start by identifying the problem type, the known quantities, the unknown target, and any assumptions or conventions being used.
- If the problem statement is ambiguous, state the ambiguity and either ask one concise clarification question or proceed under a clearly named reasonable assumption.
- Give a structured derivation, not just the final result. Show the logical chain from definitions, governing equations, identities, or principles to the conclusion.
- Explain why each important step is valid, especially substitutions, approximations, boundary conditions, sign conventions, coordinate choices, and limiting cases.
- Keep algebra explicit enough that a student can follow it. Do not skip nontrivial transformations.
- For undergraduate problems, emphasize intuition, units, diagrams described in words when helpful, and standard methods.
- For graduate problems, use more compact notation when appropriate, but still make the conceptual and mathematical logic clear.
- When multiple methods exist, choose the most direct one first. Briefly mention an alternative method if it gives useful insight.
- Check the final result using dimensions, limiting behavior, symmetry, special cases, or physical interpretation whenever possible.
- Clearly label the final answer.

Math and formatting rules:
- Always reply in clean markdown.
- Wrap inline math in $...$ and display equations in $$...$$.
- Put display equations on their own lines, separated from surrounding text by blank lines.
- Do not place display equations inside bullet points, numbered-list lines, tables, or block quotes.
- Use aligned display equations for multi-line derivations.
- Define symbols before using them unless they are standard in the user's problem.
- Avoid unsupported leaps such as 'clearly', 'obviously', or 'it is easy to show' for important steps.

Accuracy rules:
- Do not invent missing information. If a theorem, formula, or physical law is used, name it or derive it briefly.
- Track signs, constants, indices, domains, and units carefully.
- If an answer depends on convention, such as metric signature, Fourier transform normalization, or tensor index placement, state the convention.
- If you notice a likely error in the user's setup, explain it respectfully and correct it before continuing.

Tone:
- Be precise, patient, and encouraging.
- Prefer clarity over speed."""

def start_new_conversation():
    st.session_state.current_session = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    st.session_state.history_selector = st.session_state.current_session

# ---------------------------------------------------------
# STATE VARIABLE MANAGEMENT
# ---------------------------------------------------------
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = load_chat_titles()

# Step 1: Handle a fresh conversation initialization request
if "current_session" not in st.session_state or st.sidebar.button("➕ Start New Conversation"):
    st.session_state.current_session = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    st.session_state.history_selector = st.session_state.current_session

# Step 2: Sidebar History Browser
st.sidebar.title("📚 Saved Histories")
saved_sessions = list_saved_sessions()

session_options = saved_sessions.copy()
if st.session_state.current_session not in session_options:
    session_options.insert(0, st.session_state.current_session)

if session_options:
    # Match drop-down index to active state session
    current_idx = session_options.index(st.session_state.current_session)

    def display_session_title(session_id):
        return st.session_state.chat_titles.get(session_id, default_chat_title(session_id))
        
    selected_session = st.sidebar.selectbox(
        "Select Session Log",
        session_options,
        index=current_idx,
        key="history_selector",
        format_func=display_session_title,
    )
    
    # Reload historical records if selected target deviates from active working state
    if selected_session != st.session_state.current_session:
        st.session_state.current_session = selected_session
        file_path = os.path.join(HISTORY_DIR, f"{selected_session}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
        st.rerun()

    active_title = st.session_state.chat_titles.get(
        st.session_state.current_session,
        default_chat_title(st.session_state.current_session),
    )
    renamed_title = st.sidebar.text_input(
        "Chat Title",
        value=active_title,
        key=f"title_input_{st.session_state.current_session}",
    ).strip()

    if renamed_title and renamed_title != active_title:
        st.session_state.chat_titles[st.session_state.current_session] = renamed_title
        save_chat_titles(st.session_state.chat_titles)
        st.rerun()

# ---------------------------------------------------------
# USER INTERACTION & RENDERING PIPELINE
# ---------------------------------------------------------
# Display ongoing or reloaded dialogue sequences
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            render_response(msg["content"])

# Await immediate prompt interactions
if user_prompt := st.chat_input("Input problem statement or equation query..."):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Query Execution Phase
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=st.session_state.messages,
                extra_body={"thinking": {"type": "enabled"}},
                stream=False
            )
            raw_reply = response.choices[0].message.content
            
            render_response(raw_reply)
            st.session_state.messages.append({"role": "assistant", "content": raw_reply})
            
            # Immediately persist session mutations to local disk storage
            save_chat_to_disk(st.session_state.current_session, st.session_state.messages)
            
        except Exception as err:
            st.error(f"API Execution Failure: {err}")
