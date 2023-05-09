from fastai.vision.all import *
import gradio as gr

data_folder_name = 'alien_detector_data'
path = Path(data_folder_name)

learn = load_learner(path/'model.pkl')

categories = 'human', 'alien'

def classify_image(img):
	pred, idx, probs = learn.predict(img)
	return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = [f'{data_folder_name}/real_human.jpg', f'{data_folder_name}/real_alien.jpg',]

interface = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
interface.launch(inline=False)