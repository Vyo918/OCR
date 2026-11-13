"""Same-origin local browser API for private OCR collections."""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp']
LIMITS = {'max_files': 10, 'max_file_bytes': 52_428_800, 'max_batch_bytes': 104_857_600}
OVERHEAD = 1024 * 1024

class BodyCap:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST' or scope['path'] != '/api/collections':
            return await self.app(scope, receive, send)
        limit = LIMITS['max_batch_bytes'] + OVERHEAD
        length = next((v for k, v in scope['headers'] if k.lower() == b'content-length'), None)
        if length and int(length) > limit:
            response = JSONResponse({'detail': 'Upload body too large'}, status_code=413)
            return await response(scope, receive, send)
        total = 0

        async def capped_receive():
            nonlocal total
            message = await receive()
            if message['type'] == 'http.request':
                total += len(message.get('body', b''))
                if total > limit:
                    raise ValueError('Upload body too large')
            return message
        try:
            await self.app(scope, capped_receive, send)
        except ValueError as exc:
            if str(exc) != 'Upload body too large':
                raise
            response = JSONResponse({'detail': str(exc)}, status_code=413)
            await response(scope, receive, send)


class State:
    def __init__(self, workspace: Path):
        self.root = workspace / 'web'
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions = {}
        self.lock = threading.RLock()
        self.pending = []
        self.active = None
        self.stopping = False
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def worker(self):
        while True:
            with self.lock:
                if self.stopping:
                    return
                if not self.pending:
                    job = None
                else:
                    job = self.pending.pop(0)
                    job['status'], job['stage'] = 'running', 'ocr_index' if job['kind'] == 'collection' else 'answer'
                    if job.get('start'):
                        job['start']()
                    self.active = job
            if job is None:
                threading.Event().wait(.05)
                continue
            try:
                proc = subprocess.Popen([sys.executable, '-m', 'docsearch.web_jobs', job['kind'], str(job['root']), json.dumps(job['payload'])],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
                job['process'] = proc
                out, err = proc.communicate(timeout=3600)
                lines = out.splitlines()
                data = json.loads(lines[-1]) if lines else {}
                if proc.returncode or not data.get('ok'):
                    raise RuntimeError(data.get('error') or 'Processing failed. Check local server logs.')
                job['finish'](data['result'])
            except Exception as exc:
                if job.get('process') and job['process'].poll() is None:
                    os.killpg(job['process'].pid, signal.SIGKILL)
                    job['process'].wait()
                job['fail'](str(exc)[:300])
            finally:
                with self.lock:
                    self.active = None

    def close(self):
        with self.lock:
            self.stopping = True
            queued, self.pending = self.pending, []
            active = self.active
        for job in queued:
            job['fail']('Server shut down')
        if active and active.get('process') and active['process'].poll() is None:
            os.killpg(active['process'].pid, signal.SIGTERM)
        self.thread.join(timeout=5)


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        host = request.headers.get('host', '').split(':')[0].strip('[]').lower()
        if host not in {'localhost', '127.0.0.1', '::1', 'testserver'}:
            return JSONResponse({'detail': 'Invalid Host'}, status_code=400)
        origin = request.headers.get('origin')
        if origin:
            from urllib.parse import urlsplit
            parts = urlsplit(origin)
            if parts.scheme not in {'http', 'https'} or parts.hostname not in {'localhost', '127.0.0.1', '::1', 'testserver'} or parts.netloc.lower() != request.headers.get('host', '').lower():
                return JSONResponse({'detail': 'Invalid Origin'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        if request.url.path.startswith('/api/') or '/images/' in request.url.path:
            response.headers['Cache-Control'] = 'no-store'
        return response


def create_app(workspace: Path | None = None) -> FastAPI:
    app = FastAPI()
    state = State(Path(workspace or os.environ.get('DOCSEARCH_WORK', 'work')))
    app.state.web = state
    app.add_middleware(BodyCap)
    static = Path(__file__).with_name('static')
    app.mount('/static', StaticFiles(directory=static, check_dir=False), name='static')

    @app.on_event('shutdown')
    def shutdown():
        state.close()

    def session(request: Request):
        sid = request.cookies.get('docsearch_session')
        info = state.sessions.get(sid)
        if not info:
            from fastapi import HTTPException
            raise HTTPException(401, 'Session required')
        return sid, info

    def csrf(request: Request, token: str | None):
        sid, info = session(request)
        if not token or not secrets.compare_digest(token, info['csrf']):
            from fastapi import HTTPException
            raise HTTPException(403, 'Invalid CSRF token')
        return sid, info

    @app.get('/')
    def index():
        file = static / 'index.html'
        if not file.is_file():
            return HTMLResponse('Web interface is not installed', status_code=503)
        return FileResponse(file)

    @app.get('/api/session')
    def bootstrap(request: Request):
        sid = request.cookies.get('docsearch_session')
        info = state.sessions.get(sid) if sid else None
        if info is None:
            sid, info = secrets.token_urlsafe(32), {'csrf': secrets.token_urlsafe(32), 'collections': {}}
            state.sessions[sid] = info
        response = JSONResponse({'csrf_token': info['csrf'], 'limits': LIMITS, 'supported_extensions': EXTENSIONS})
        response.set_cookie('docsearch_session', sid, httponly=True, samesite='strict', secure=request.url.scheme == 'https', path='/')
        return response

    @app.post('/api/collections')
    async def upload(request: Request, x_csrf_token: str | None = Header(None)):
        sid, info = csrf(request, x_csrf_token)
        try:
            form = await request.form(max_files=LIMITS['max_files'], max_fields=LIMITS['max_files'], max_part_size=1024 * 1024)
        except Exception:
            from fastapi import HTTPException
            raise HTTPException(400, 'Malformed multipart upload')
        if set(form.keys()) != {'files'} or not all(isinstance(x, UploadFile) for x in form.getlist('files')):
            from fastapi import HTTPException
            raise HTTPException(422, 'Multipart upload must contain only files')
        files = form.getlist('files')
        if not 1 <= len(files) <= LIMITS['max_files']:
            from fastapi import HTTPException
            raise HTTPException(422, 'Upload between 1 and 10 files')
        cid = uuid.uuid4().hex
        root = state.root / sid / cid
        uploads = root / 'uploads'
        uploads.mkdir(parents=True)
        metadata, paths, total = [], [], 0
        for index, upload in enumerate(files):
            name = Path(upload.filename or '').name
            ext = Path(name).suffix.lower()
            if ext not in EXTENSIONS:
                from fastapi import HTTPException
                raise HTTPException(422, f'Unsupported file: {name}')
            data = await upload.read(LIMITS['max_file_bytes'] + 1)
            total += len(data)
            if not data or len(data) > LIMITS['max_file_bytes'] or total > LIMITS['max_batch_bytes']:
                from fastapi import HTTPException
                raise HTTPException(413 if data else 422, 'Empty or oversized upload')
            if ext == '.pdf':
                if not data.startswith(b'%PDF-') or b'/Encrypt' in data[:65536]:
                    from fastapi import HTTPException
                    raise HTTPException(422, 'Invalid or encrypted PDF')
            else:
                try:
                    from PIL import Image
                    import io
                    with Image.open(io.BytesIO(data)) as im:
                        if im.width * im.height > 100_000_000:
                            raise ValueError()
                        im.verify()
                except Exception:
                    from fastapi import HTTPException
                    raise HTTPException(422, f'Invalid image: {name}')
            path = uploads / f'{index:02d}{ext}'
            path.write_bytes(data)
            metadata.append({'name': name, 'size': len(data)})
            paths.append(str(path))
        record = {'id': cid, 'files': metadata, 'status': 'queued', 'stage': 'queued', 'error': None}
        info['collections'][cid] = {'record': record, 'root': root, 'questions': {}}
        state.pending.append({'kind':'collection','root':root,'payload':{'files':paths},'start':lambda:record.update(status='running',stage='ocr_index'),'finish':lambda _: record.update(status='ready',stage='ready'),'fail':lambda e: record.update(status='failed',stage='failed',error=e)})
        return JSONResponse(record, status_code=202)

    @app.get('/api/collections/{collection_id}')
    def get_collection(request: Request, collection_id: str):
        _, info = session(request)
        col = info['collections'].get(collection_id)
        if not col:
            from fastapi import HTTPException
            raise HTTPException(404, 'Collection not found')
        return col['record']

    @app.post('/api/collections/{collection_id}/questions')
    async def question(request: Request, collection_id: str, x_csrf_token: str | None = Header(None)):
        sid, info = csrf(request, x_csrf_token)
        col = info['collections'].get(collection_id)
        if not col:
            from fastapi import HTTPException
            raise HTTPException(404, 'Collection not found')
        if col['record']['status'] != 'ready':
            from fastapi import HTTPException
            raise HTTPException(409, 'Collection is not ready')
        try:
            body = await request.json()
            text = body.get('question', '').strip()
        except Exception:
            text = ''
        if not text or len(text) > 2000:
            from fastapi import HTTPException
            raise HTTPException(422, 'Question must contain 1 to 2000 characters')
        jid = uuid.uuid4().hex
        job = {'id':jid,'question':text,'status':'queued','stage':'queued','error':None,'result':None}
        col['questions'][jid] = job
        def finish(data):
            images = []
            image_map = {}
            for p in data.get('images', []):
                path = Path(p).resolve()
                dest = (state.root / sid / collection_id / 'questions' / jid / path.name).resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)
                if path != dest:
                    import shutil
                    shutil.copyfile(path, dest)
                iid = uuid.uuid4().hex
                image_map[iid] = dest
                images.append({'name': path.name, 'url': f'/api/collections/{collection_id}/questions/{jid}/images/{iid}'})
            job.update(status='ready',stage='ready',result={k:data[k] for k in ('answer','queries','evidence') } | {'images':images})
            job['image_map'] = image_map
        state.pending.append({'kind':'question','root':col['root'],'payload':{'question':text,'session':sid,'collection':collection_id,'job':jid},'start':lambda:job.update(status='running',stage='answer'),'finish':finish,'fail':lambda e:job.update(status='failed',stage='failed',error=e)})
        return JSONResponse(job, status_code=202)

    @app.get('/api/collections/{collection_id}/questions/{question_id}')
    def get_question(request: Request, collection_id: str, question_id: str):
        _, info = session(request)
        job = info['collections'].get(collection_id, {}).get('questions', {}).get(question_id)
        if not job:
            from fastapi import HTTPException
            raise HTTPException(404, 'Question not found')
        return {k:v for k,v in job.items() if k != 'image_map'}

    @app.get('/api/collections/{collection_id}/questions/{question_id}/images/{image_id}')
    def image(request: Request, collection_id: str, question_id: str, image_id: str):
        _, info = session(request)
        job = info['collections'].get(collection_id, {}).get('questions', {}).get(question_id)
        path = job.get('image_map', {}).get(image_id) if job else None
        if not path:
            from fastapi import HTTPException
            raise HTTPException(404, 'Image not found')
        path = Path(path).resolve()
        if not path.is_relative_to(state.root.resolve()) or not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}:
            from fastapi import HTTPException
            raise HTTPException(404, 'Image not found')
        return FileResponse(path)

    return app
