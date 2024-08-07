from fastapi import FastAPI
from pydantic import BaseModel
from machine_learning_model import preferences_vector_multiplication
import requests
from PIL import Image
from io import BytesIO
from pyxios import Pyxios


app = FastAPI()

class postCharacteristics(BaseModel):
    post_id: str
    text: str
    img: str
    token_bearer: str

@app.get('/')
def test_root():
    return {"Hello World"}

@app.post('/create-vector-classificator')
def create_vector(body: postCharacteristics):
    print(body.text)
    print(body.img)
    print(body.token_bearer)
    
    response = requests.get(body.img)

    image_to_classify = Image.open(BytesIO(response.content))

    vector = preferences_vector_multiplication(body.text, image_to_classify)

    Pyxios(f'http://localhost:80/notes/update-note/{body.post_id}').patch( 
        headers = {
            'Authorization': f'Bearer {body.token_bearer}'
          },
        json= {
            vector
          })
  