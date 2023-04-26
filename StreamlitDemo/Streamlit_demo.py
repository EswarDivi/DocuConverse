import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from streamlit_chat import message
import random


# Set page title
st.set_page_config(page_title="Demo Streamlit App")

# Create sidebar
st.sidebar.title("Menu")
options = [
    "Displaying Text",
    "Data Elements",
    "Media Elements",
    "Interactive Input Elements",
    "Chart Elements",
    "Progress and Status Elements",
    "StreamlitChat",
]
choice = st.sidebar.radio("Select an option", options)


if choice == "Displaying Text":
    st.write("Streamlit  Demo")
    st.code("st.text()", language="python")

    st.header("This is Heading 1 in Markdown")
    st.code("st.markdown()", language="python")

    st.title("This is a title")
    st.code("st.title()", language="python")

    st.header("Header")
    st.code("st.header()", language="python")

    st.subheader("Sub Header")
    st.code("st.subheader()", language="python")

    st.latex(r"x^2 + y^2 = z^2")
    st.code("st.latex()", language="python")

    st.write("Streamlit can display a lot of other things too!")
    st.code("st.write()", language="python")

    st.divider()
    st.subheader("Above is a divider")
    st.code("st.divider()", language="python")

elif choice == "Data Elements":
    st.header("Data Elements")
    st.subheader("DataFrame")
    df = pd.DataFrame(
        np.random.randn(50, 20), columns=("col %d" % i for i in range(20))
    )
    st.dataframe(df)

    st.header("Metrics is also Possible")

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")

    st.header("Json format")
    st.json(
        {
            "foo": "bar",
            "baz": "boz",
            "stuff": [
                "stuff 1",
                "stuff 2",
                "stuff 3",
                "stuff 5",
            ],
        }
    )
elif choice == "Media Elements":
    st.header("Media Elements")
    st.subheader("Image")
    st.image(
        "https://images.unsplash.com/photo-1579353977828-2a4eab540b9a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8c2FtcGxlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60",
        caption="This is an example image.",
        use_column_width=True,
    )

    st.subheader("This is Youtube Video")
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


elif choice == "Interactive Input Elements":
    st.header("Interactive Input Elements")
    name = st.text_input("Enter your name", "I didn't Know")
    st.write("Hello, " + name + "!")

    A = st.button("Click here")
    if A:
        st.write("You Clicked It")
    st.checkbox("Check this")
    st.radio("Radio", [1, 2, 3])
    st.selectbox("Select", [1, 2, 3])
    st.multiselect("Multiple selection", [21, 85, 53])
    st.slider("Slide", min_value=10, max_value=20)
    st.select_slider("Slide to select", options=[1, 2, 3, 4])
    st.text_input("Enter some text")
    st.number_input("Enter a number")
    st.date_input("Date input")
    st.time_input("Time input")
    st.file_uploader("File uploader")
    st.color_picker("Color Picker")

elif choice == "Chart Elements":
    st.header("Chart Elements")

    st.header("Line Chart")
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    st.line_chart(df)

    st.header("Plotly ")

    df = px.data.gapminder()

    fig = px.scatter(
        df.query("year==2007"),
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
    )

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    st.write("Tabs are Supported WoW!!")
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)

    st.header("Map")

    df = pd.DataFrame(
        np.random.randn(100, 2) / [50, 50] + [11.016844, 76.955833],
        columns=["lat", "lon"],
    )

    st.map(df)


elif choice == "Progress and Status Elements":
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    st.spinner()
    with st.spinner(text="This is Spinning"):
        time.sleep(2)
        st.success("Done")
    st.balloons()
    st.error("Error message")
    st.warning("Warning message")
    st.info("Info message")
    st.success("Success message")
    e = RuntimeError("This is an exception of type RuntimeError which never Happended")
    st.exception(e)

elif choice == "StreamlitChat":
    # accept input from user
    user_input = st.text_input("You:", key="input")
    if user_input:
        message(f"You: {user_input}", is_user=True)
        random_number = random.randint(0, 100)
        message(
            f"Bot: Hi there! I think You Lucky Number is {random_number}", is_user=False
        )
