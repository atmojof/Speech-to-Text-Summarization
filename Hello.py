import streamlit as st

st.set_page_config(page_title="Hello",page_icon="👋",layout="wide")

st.write("# Hey there! Welcome aboard the Population Calculator App!  👋")

st.markdown(
    """
        Let's dive into it, shall we?
        ### Introduction
        **We've got a nifty collection of population data from Statistics Indonesia specifically focusing on Jabodetabek**
        - Scope: We're talking about the whole Jabodetabek here—Jakarta, Bogor, Depok, Bekasi
        ### How to use this app
        1. **Population Estimator**: Just punch in your desired location, and voila! Watch the population estimation magic unfold.
        2. **Travel Time**: Need to know the catchment area of a spot based on travel method? Easy peasy! Just enter the coordinates and select your mode of travel—driving, walking, or biking and see the population estimation as well.
        3. **POI Analysis**: Discover key Points of Interest (POIs) such as schools, hospitals, secondary property market (Landed Residential, Condominium, Shophouse, Land Lot) and more, all at your fingertips! 
        """
)

#import streamlit as st
#from hashlib import sha256
#
## Set page config
#st.set_page_config(page_title="Hello", page_icon="👋", layout="wide")
#
## Define usernames and hashed passwords
#usernames = ['user1', 'user2', 'user3']
#passwords = ['password1', 'password2', 'password3']
#hashed_passwords = [sha256(p.encode()).hexdigest() for p in passwords]
#
## Store users in a dictionary with hashed passwords
#credentials = dict(zip(usernames, hashed_passwords))
#
## Verify login function
#def verify_login(username, password):
#    hashed_input = sha256(password.encode()).hexdigest()
#    return credentials.get(username) == hashed_input
#
## Initialize login state in session
#if "logged_in" not in st.session_state:
#    st.session_state.logged_in = False
#
## Login page
#if not st.session_state.logged_in:
#    st.title("Login Page")
#    
#    username = st.text_input("Username")
#    password = st.text_input("Password", type="password")
#    
#    if st.button("Login"):
#        if verify_login(username, password):
#            st.session_state.logged_in = True
#            st.success("Login successful! Redirecting...")
#            st.experimental_rerun()
#        else:
#            st.error("Invalid username or password")
#else:
#
#    st.markdown(
#        """
#
#            Let's dive into it, shall we?
#
#            ### Introduction
#            **We've got a nifty collection of population data from Statistics Indonesia specifically focusing on Jabodetabek**
#            - Scope: We're talking about the whole Jabodetabek here—Jakarta, Bogor, Depok, Bekasi
#            ### How to use this app
#
#            1. **Population Estimator**: Just punch in your desired location, and voila! Watch the population estimation magic unfold.
#            2. **Travel Time**: Need to know the catchment area of a spot based on travel method? Easy peasy! Just enter the coordinates and select your mode of travel—driving, walking, or biking and see the population estimation as well.
#            3. **POI Analysis**: Discover key Points of Interest (POIs) such as schools, hospitals, secondary property market (Landed Residential, Condominium, Shophouse, Land Lot) and more, all at your fingertips! 
#
#            """
#    )