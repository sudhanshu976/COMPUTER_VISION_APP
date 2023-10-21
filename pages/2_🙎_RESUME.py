from pathlib import Path

import streamlit as st
from PIL import Image
st.set_page_config(
    page_title="CV WEB APP"
)


# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "Resume-Sudhanshu-Gusain (3).pdf"
profile_pic = current_dir / "assets" / "profile-pic.png"


# --- GENERAL SETTINGS ---
PAGE_TITLE = "Digital CV | Sudhanshu"
PAGE_ICON = ":wave:"
NAME = "SUDHANSHU"
DESCRIPTION = """
Aspiring Data Scientist | 18-Year-Old Data Enthusiast | 1 Year of Hands-On Experience | Passionate about Solving Real-World Problems"
"""
EMAIL = "gusainsudhanshu43@gmail.com"
SOCIAL_MEDIA = {
    "NLP - Website": "https://nlpappbysudhanshu.streamlit.app",
    "LinkedIn": "https://www.linkedin.com/in/sudhanshu-gusain976/",
    "GitHub": "https://github.com/sudhanshu976",
    "CV - Website": "https://cvappbysudhanshu.streamlit.app",
}
PROJECTS = {
    "🏆 MALARIA DIAGNOSIS MODEL",
    "🏆 POTATO DISEASE CLASSIFIER",
    "🏆 ",
}




# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_pic, width=230)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )
    st.write("📫", EMAIL)


# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")


# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Experience & Qulifications")
st.write(
    """
- ✔️ 1 Year expereince of performing various Data Science and NLP / CV tasks
- ✔️ Strong hands on experience and knowledge in Python , ML , DL , NLP and CV
- ✔️ Good understanding of statistical principles and their respective applications
- ✔️ Excellent team-player and displaying strong sense of initiative on tasks
"""
)


# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write(
    """
- 👩‍💻 Programming: Python (Scikit-learn, Pandas , Numpy , Pytorch , Tensorflow)
- 📊 Data Visulization: PowerBi, Matplotlib , Seaborn
- 📚 Modeling: Supervised and Unsupervised ML algorithms , ANN , RNN , CNN
- 🗄️ Databases: MySQL
- 🗄️ WEB DEPLOYMENT: FLASK , Streamlit , Heroku
"""
)


# --- WORK HISTORY ---
st.write('\n')
st.subheader("Work History")
st.write("---")

# --- JOB 1
st.write("🚧", "**Freelancer Data Scientist and CV Engineer**")
st.write("05/2023 - Present")
st.write(
    """
- ► Used PowerBI for creating interactive dashboards 
- ► Solved many ML , DL and NLP and CV problems in various fields like medical , agriculture , etc
- ► Well versed in solving real life problems especially using CV
"""
)

# # --- Projects & Accomplishments ---
# st.write('\n')
# st.subheader("Projects & Accomplishments")
# st.write("---")
# for project, link in PROJECTS.items():
#     st.write(f"[{project}]({link})")