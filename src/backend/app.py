from fastapi import FastAPI, Path, Query, HTTPException, status, Depends
from typing import Optional, List, Annotated
from pydantic import BaseModel
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import models
app = FastAPI()

models.Base.metadata.create_all(bind=engine)

""" class Item(BaseModel):
    name:str
    price:float
    brand: Optional[str] = None

class UpdateItem(BaseModel):
    name:Optional[str] = None
    price:Optional[float] = None
    brand: Optional[str] = None """

class Document(BaseModel):
    filename: str
    ext: str
    createdate: str

class conversation(BaseModel):
    question: str
    answer: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

""" inventory = {}
#path parameter
@app.get("/get-item/{item_id}")
def get_item(item_id: int = Path(description="The ID of the item")):
    if item_id not in inventory:
        #return {"Error":"Item does not exist"}
        raise HTTPException(status_code=404, detail="Item does not Exist")
    return inventory[item_id]
#path parameter
@app.get("/get-by-name/{name}")
def get_by_name(name: str = None):
    for item_id in inventory:
        if inventory[item_id].name == name:
            return inventory[item_id]
        else:
            return {"Data":"Not Found"}
#query parameter
@app.get("/get-by-name")
def get_by_name(name: Optional[str] = None):
    for item_id in inventory:
        if inventory[item_id].name == name:
            return inventory[item_id]
        else:
            return {"Data":"Not Found"}
        
@app.post("/create-item/{item_id}")
def create_item(item_id: int, item: Item):
    if item_id in inventory:
        return {"Error": "Item ID already exists."}
    inventory[item_id] = item
    return inventory[item_id]

@app.put("/update-item/{item_id}")
def update_item(item_id: int, item: UpdateItem):
    if item_id not in inventory:
        return {"Data":"Item does not exist"}
    if item.name != None:
        inventory[item_id].name = item.name
    if item.price != None:
        inventory[item_id].price = item.price
    if item.brand != None:
        inventory[item_id].brand = item.brand
    return inventory[item_id]

@app.delete("/delete-item")
def delete_item(item_id: int = Query(..., description="The ID data will be deleted")):
    if item_id not in inventory:
        return {"Error":"ID Does not exist"}
    del inventory[item_id] """

@app.post("/document")
async def create_document(document: Document, db: db_dependency):
    db_filename = models.Document(filename=document.filename)
    db_createdate = models.Document(createdate=document.createdate)
    db.add(db_filename)
    db.add(db_createdate)
    db.commit()
    db.refresh(db_filename,db_createdate)

