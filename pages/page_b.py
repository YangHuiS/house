import streamlit as st
from multipage_streamlit import State
import matplotlib.pyplot as plt

def run():
    state = State(__name__)
    # the above line is required if you want to save states across page switches.
    # you can provide your own namespace prefix to make keys unique across pages.
    # here we use __name__ for convenience.
    # st.header("Page B")
    st.snow()
    st.balloons()
    # st.write('敬请期待')
    j = plt.imread('qi.jpg')
    st.image(j)
    # here's the "magic": state(name, default, ...) returns the namespace-prefixed
    # key name. if a previously saved state exist, the widget is set to it. if not,
    # the widget is set to default if it is specified.

    state.save()  # MUST CALL THIS TO SAVE THE STATE!