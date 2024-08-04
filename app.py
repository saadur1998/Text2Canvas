""" 
    This file is the main file of the project.
"""

# imports
import streamlit as st
from text_to_image import generate_image
from feature_to_sprite import generate_sprites


def setup():
    """
    Streamlit related setup. This has to be run for each page.
    """

    # hide hamburger menu
    hide_streamlit_style = """
    
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def main():
    """
    Main function of the app.
    """

    setup()

    # title, subheader, and description
    st.title("Text2Canvas")

    st.subheader("A tool that demonstrates the power of Diffusion")

    # horizontal line and line break
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select a mode", ["Home", "Text2Image", "Feature2Sprite"])
    # st.sidebar.write("Select a mode to get started")

    # main content

    if mode == "Home":
        st.write(
            """
        This tool is a demonstration of the power of Diffusion. It helps you generate images from text. There are two modes:
        1. **Text2Image**: This mode generates high quality image based on a given prompt. It uses a pretrained model.
        2. **Feature2Sprite**: This mode generates 16*16 images of sprites based on a combination of features. It uses a custom model trained on a dataset of sprites.
        
        To get started, select a mode from the sidebar.
        """
        )

    elif mode == "Text2Image":
        st.write(
            """
        This mode generates high quality image based on a given prompt. It uses a pretrained model from huggingface.
        """
        )

        form = st.form(key="my_form")

        prompt = form.text_input("Enter a prompt", value="A painting of a cat")

        submit_button = form.form_submit_button(label="Generate")

        if submit_button:
            st.write("Generating image...")
            image = generate_image(prompt)
            st.image(image, caption="Generated Image", use_column_width=True)

    elif mode == "Feature2Sprite":
        st.write(
            """
        This mode generates 16*16 images of sprites based on a combination of features. It uses a custom model trained on a dataset of sprites.
        """
        )

        form = st.form(key="my_form")

        # add sliders
        hero = form.slider("Hero", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
        non_hero = form.slider(
            "Non Hero", min_value=0.0, max_value=1.0, value=0.0, step=0.01
        )
        food = form.slider("Food", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        spell = form.slider("Spell", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        side_facing = form.slider(
            "Side Facing", min_value=0.0, max_value=1.0, value=0.0, step=0.01
        )
        # add submit button
        submit_button = form.form_submit_button(label="Generate")

        # create feature vector
        if submit_button:
            feature_vector = [hero, non_hero, food, spell, side_facing]
            # show loader
            with st.spinner("Generating sprite..."):
                # horizontal line and line break
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.subheader("Your Sprite")
                st.markdown("<br>", unsafe_allow_html=True)

                _ = generate_sprites(feature_vector)


if __name__ == "__main__":
    main()
