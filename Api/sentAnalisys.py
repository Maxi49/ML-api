from fastapi import FastAPI
from pydantic import BaseModel
from machine_learning_model import preferences_vector_multiplication
from PIL import Image
from io import BytesIO
from typing import Dict
import json
import aiohttp

app = FastAPI()

class postCharacteristics(BaseModel):
    data: Dict

@app.get('/')
def test_root():
    return {"Hello World"}
    
@app.post('/create-vector-classificator')
async def create_vector(req: dict):
    data = req['Records']
    receiptHandle = data[0]
    
    messageDeleteId = receiptHandle['receiptHandle']

    body = json.loads(data[0]['body'])
    print(body)

    token = body['token']
    post_id = body['postId']

    async with aiohttp.ClientSession() as session:
        async with session.get(body['imgUrl']) as response:
            if response.status == 200:
                image_data = await response.read()
                image_to_classify = Image.open(BytesIO(image_data))
                vector = preferences_vector_multiplication(body['text'], image_to_classify)

                update_url = f'http://localhost:80/vector/create-vector'
                headers = {"Authorization": token, "Content-Type": "application/json; charset=utf-8"}
                vector_to_send = {"data":{ "vector": vector.tolist(), "note": post_id, "receiptHandle": messageDeleteId }}

                async with session.post(update_url, json=vector_to_send, headers=headers) as update_response:
                  if update_response.status == 201:
                    print("Solicitud patch exitosa")
                    return {"status": 201, "success": True, "message": "Procesamiento exitoso"}
                  else:
                      return {"statusCode": 400, "success": False}
            else:
                print(f"Error en la solicitud de imagen: {response.status}")

    return {"message": "Procesamiento exitoso"}
