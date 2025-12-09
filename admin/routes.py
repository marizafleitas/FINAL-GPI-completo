from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
import subprocess

router = APIRouter()
templates = Jinja2Templates(directory="admin/templates")

DOCS_DIR = "docs"


@router.get("/", response_class=HTMLResponse)
async def admin_home(request: Request):
    return templates.TemplateResponse("admin_index.html", {"request": request})


@router.get("/list", response_class=HTMLResponse)
async def list_docs(request: Request, msg: str = None):
    files = os.listdir(DOCS_DIR)
    data = []
    for f in files:
        path = os.path.join(DOCS_DIR, f)
        if os.path.isfile(path):
            info = {
                "name": f,
                "size": round(os.path.getsize(path) / 1024, 2)
            }
            data.append(info)

    return templates.TemplateResponse("admin_list.html", {
        "request": request,
        "files": data,
        "msg": msg
    })


@router.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("admin_upload.html", {"request": request})


@router.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    save_path = os.path.join(DOCS_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    subprocess.run(["python", "procesar_pdfs.py"])

    # REDIRECCIÓN con mensaje
    return RedirectResponse(
        url=f"/admin/list?msg={file.filename} subido correctamente.",
        status_code=303
    )


@router.get("/replace/{filename}", response_class=HTMLResponse)
async def replace_form(request: Request, filename: str):
    return templates.TemplateResponse("admin_replace.html", {
        "request": request,
        "filename": filename
    })


@router.post("/replace/{filename}")
async def replace_pdf(request: Request, filename: str, file: UploadFile = File(...)):
    save_path = os.path.join(DOCS_DIR, filename)

    # eliminar el anterior
    if os.path.exists(save_path):
        os.remove(save_path)

    # guardar el nuevo
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    subprocess.run(["python", "procesar_pdfs.py"])

    return RedirectResponse(
        url=f"/admin/list?msg={filename} reemplazado correctamente.",
        status_code=303
    )


@router.get("/delete/{filename}")
async def delete_pdf(request: Request, filename: str):
    path = os.path.join(DOCS_DIR, filename)

    if os.path.exists(path):
        os.remove(path)
        subprocess.run(["python", "procesar_pdfs.py"])
        msg = f"{filename} eliminado correctamente."
    else:
        msg = f"{filename} no encontrado."

    return RedirectResponse(url=f"/admin/list?msg={msg}", status_code=303)
