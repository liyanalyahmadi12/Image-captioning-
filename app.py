# app.py

import gradio as gr
from PIL import Image
from image_cap import generate_caption_from_pil


def caption_from_gradio(image):
    if image is None:
        return "Please upload an image."
    # Gradio gives you a NumPy array; convert to PIL
    pil_image = Image.fromarray(image)
    return generate_caption_from_pil(pil_image)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🖼️ Simple Image Captioning (BLIP)
        Upload an image and get a short caption.
        """
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(
                label="Upload image",
                type="numpy"
            )
            btn = gr.Button("Generate caption")

        with gr.Column():
            caption_output = gr.Textbox(
                label="Caption",
                lines=2,
                interactive=False
            )

    btn.click(fn=caption_from_gradio, inputs=img_input, outputs=caption_output)

demo.launch()
