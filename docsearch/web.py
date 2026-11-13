"""Same-origin local browser API for private OCR collections."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

EXTENSIONS = ('.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
RASTER_EXTENSIONS = frozenset(EXTENSIONS[1:])
LOOPBACK_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})
DNS_NAME = re.compile(r'[a-z0-9-]+(?:\.[a-z0-9-]+)*')
LIMITS = {'max_files': 10, 'max_file_bytes': 52_428_800, 'max_batch_bytes': 104_857_600}
BODY_LIMIT = LIMITS['max_batch_bytes'] + 1024 * 1024
PUBLIC_ERRORS = {
    'timeout': 'Processing took too long.', 'pdf': 'The PDF is invalid, encrypted, too large, or unreadable.',
    'image': 'The image is invalid or too large.', 'pipeline': 'Processing failed. Check the local server log.',
    'shutdown': 'Server shut down before processing completed.',
}


class BodyCap:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST' or scope['path'] != '/api/collections':
            return await self.app(scope, receive, send)
        length = next((v for k, v in scope['headers'] if k.lower() == b'content-length'), None)
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                response = JSONResponse({'detail': 'Malformed Content-Length'}, status_code=400)
                return await response(scope, receive, send)
            if declared < 0:
                response = JSONResponse({'detail': 'Malformed Content-Length'}, status_code=400)
                return await response(scope, receive, send)
            if declared > BODY_LIMIT:
                response = JSONResponse({'detail': 'Upload body too large'}, status_code=413)
                return await response(scope, receive, send)
        total = 0
        exceeded = False

        async def capped_receive():
            nonlocal total, exceeded
            message = await receive()
            if message['type'] == 'http.request':
                total += len(message.get('body', b''))
                if total > BODY_LIMIT:
                    exceeded = True
                    scope.setdefault('state', {})['body_cap_exceeded'] = True
                    raise BodyTooLarge
            return message

        try:
            await self.app(scope, capped_receive, send)
        except BodyTooLarge:
            response = JSONResponse({'detail': 'Upload body too large'}, status_code=413)
            await response(scope, receive, send)


class BodyTooLarge(Exception):
    pass


class State:
    def __init__(self, workspace: Path, process_args=None):
        self.root = workspace / 'web'
        self.process_args = process_args or (lambda job: [sys.executable, '-m', 'docsearch.web_jobs', job['kind'], str(job['root']), json.dumps(job['payload'])])
        self.sessions = {}
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.pending = []
        self.active = None
        self.stopping = False
        self.thread = None

    def start(self):
        with self.condition:
            if self.thread is None:
                self.thread = threading.Thread(target=self.worker, name='docsearch-web-jobs', daemon=True)
                self.thread.start()

    def enqueue(self, job):
        with self.condition:
            if self.stopping:
                raise HTTPException(503, 'Server is shutting down')
            self.pending.append(job)
            self.condition.notify()

    def _stop_process(self, process):
        if process.poll() is not None:
            process.wait()
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def worker(self):
        while True:
            with self.condition:
                while not self.pending and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                job = self.pending.pop(0)
                self.active = job
                job['start']()
                try:
                    proc = subprocess.Popen(
                        self.process_args(job), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, start_new_session=True, env={**os.environ, 'DOCSEARCH_PROGRESS_FILE': str(job['progress_path'])},
                    )
                except Exception as exc:
                    diagnostic = repr(exc)
                    log_path = job.get('log_path', job['root'] / 'job.log')
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(diagnostic, encoding='utf-8')
                    job['fail']('pipeline', diagnostic)
                    self.active = None
                    continue
                job['process'] = proc
            try:
                out, err = proc.communicate(timeout=3600)
                log_path = job.get('log_path', job['root'] / 'job.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f'stdout:\n{out}\nstderr:\n{err}', encoding='utf-8')
                lines = out.rstrip('\n').split('\n')
                try:
                    data = json.loads(lines[-1]) if lines else {}
                except (ValueError, IndexError):
                    data = {}
                with self.condition:
                    if self.stopping:
                        job['fail']('shutdown', 'active during shutdown')
                    elif proc.returncode or not data.get('ok'):
                        job['fail'](data.get('category', 'pipeline'), data.get('diagnostic', err or out))
                    else:
                        job['finish'](data['result'])
            except subprocess.TimeoutExpired:
                self._stop_process(proc)
                out, err = proc.communicate()
                log_path = job.get('log_path', job['root'] / 'job.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f'timeout\nstdout:\n{out}\nstderr:\n{err}', encoding='utf-8')
                job['fail']('timeout', 'processing exceeded 3600 seconds')
            except Exception as exc:
                if proc.poll() is None:
                    self._stop_process(proc)
                job['fail']('pipeline', repr(exc))
            finally:
                with self.condition:
                    self.active = None
                    self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopping = True
            queued, self.pending = self.pending, []
            active = self.active
            for job in queued:
                job['fail']('shutdown', 'queued during shutdown')
            self.condition.notify_all()
        if active and active.get('process'):
            self._stop_process(active['process'])
        if self.thread and threading.current_thread() is not self.thread:
            self.thread.join()


class SecurityHeaders(BaseHTTPMiddleware):
    def __init__(self, app, allowed_hosts=LOOPBACK_HOSTS):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts

    async def dispatch(self, request, call_next):
        host = request.headers.get('host', '')
        try:
            permitted_host = _parse_authority(host, self.allowed_hosts)
        except ValueError:
            return _security_response({'detail': 'Invalid Host'}, 400, request.url.path)
        if permitted_host is None:
            return _security_response({'detail': 'Invalid Host'}, 400, request.url.path)
        origin = request.headers.get('origin')
        if origin is not None:
            try:
                parts = urlsplit(origin)
                origin_host = _parse_authority(parts.netloc, self.allowed_hosts)
                valid_origin = (parts.scheme == 'http' and origin_host is not None and
                                origin_host == permitted_host and not parts.username and not parts.password and
                                parts.path in ('', '/') and not parts.query and not parts.fragment)
            except ValueError:
                valid_origin = False
            if not valid_origin:
                return _security_response({'detail': 'Invalid Origin'}, 403, request.url.path)
        response = await call_next(request)
        _add_security_headers(response, request.url.path)
        return response


def _parse_authority(authority: str, allowed_hosts=LOOPBACK_HOSTS):
    if not authority or authority != authority.strip() or any(c in authority for c in '/?#@\\'):
        raise ValueError('invalid authority')
    try:
        parsed = urlsplit('//' + authority)
        hostname = _normalize_host(parsed.hostname or '')
        port = parsed.port
    except ValueError as exc:
        raise ValueError('invalid authority') from exc
    if parsed.netloc != authority or hostname not in allowed_hosts:
        return None
    if port is not None and not 1 <= port <= 65535:
        return None
    if ':' in hostname and not authority.startswith('['):
        return None
    return hostname, port


def _normalize_host(hostname: str) -> str:
    if ':' in hostname:
        try:
            return str(ipaddress.ip_address(hostname))
        except ValueError:
            return hostname
    return hostname


def _allowed_hosts(entries):
    allowed = set(LOOPBACK_HOSTS)
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError('allowed host must be a string')
        item = entry.strip().lower()
        if not item:
            raise ValueError('empty allowed host')
        try:
            allowed.add(str(ipaddress.ip_address(item)))
            continue
        except ValueError:
            pass
        if not DNS_NAME.fullmatch(item):
            raise ValueError(f'invalid allowed host: {entry!r}')
        allowed.add(item)
    return frozenset(allowed)




def _add_security_headers(response, path):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    if path.startswith('/api/') or '/images/' in path:
        response.headers['Cache-Control'] = 'no-store'


def _security_response(body, status, path):
    response = JSONResponse(body, status_code=status)
    _add_security_headers(response, path)
    return response


def _failure(category, diagnostic):
    return {'status': 'failed', 'stage': 'failed', 'error': PUBLIC_ERRORS.get(category, PUBLIC_ERRORS['pipeline'])}


def _contained_file(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if not path.is_relative_to(directory) or path.suffix.lower() not in RASTER_EXTENSIONS or not path.is_file():
        raise ValueError('job output escaped or is not a raster image')
    return path


def _timed(record, progress_path, kind, clock):
    from .progress import percent_for, read_progress
    now = clock()
    started, finished = record['started'], record['finished']
    elapsed_end = finished if finished is not None else now
    run = None if started is None else round(max(0, elapsed_end - started), 1)
    queued = round(max(0, (started if started is not None else now) - record['submitted']), 1)
    item = read_progress(progress_path)
    measured_stage = item['stage'] if item and record['status'] in ('running', 'failed') else record['stage']
    stage = record['stage']
    done, total = (item['done'], item['total']) if item else (None, None)
    percent = 100 if record['status'] == 'ready' else percent_for(kind, measured_stage, done, total)
    public = {key: value for key, value in record.items() if key not in ('image_map', 'submitted', 'started', 'finished')}
    return {**public, 'run_seconds': run, 'queued_seconds': queued,
            'progress': {'stage': stage, 'percent': percent, 'done': done, 'total': total, 'unit': item['unit'] if item else None}}


def create_app(workspace: Path | None = None, process_args=None, allowed_hosts=None, clock=time.monotonic) -> FastAPI:
    if allowed_hosts is None:
        allowed_hosts = [item for item in os.environ.get('DOCSEARCH_ALLOWED_HOSTS', '').split(',') if item.strip()]
    allowed = _allowed_hosts(allowed_hosts)
    state = State(Path(workspace or os.environ.get('DOCSEARCH_WORK', 'work')), process_args)
    static = Path(__file__).with_name('static')

    @asynccontextmanager
    async def lifespan(app):
        state.start()
        try:
            yield
        finally:
            state.close()

    app = FastAPI(lifespan=lifespan)
    app.state.web = state
    app.add_middleware(BodyCap)
    app.add_middleware(SecurityHeaders, allowed_hosts=allowed)
    app.mount('/static', StaticFiles(directory=static, check_dir=False), name='static')

    def session(request: Request):
        sid = request.cookies.get('docsearch_session')
        info = state.sessions.get(sid)
        if not info:
            raise HTTPException(401, 'Session required')
        return sid, info

    def csrf(request: Request, token: str | None):
        sid, info = session(request)
        if not token or not secrets.compare_digest(token, info['csrf']):
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
        with state.lock:
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
        except BodyTooLarge as exc:
            raise HTTPException(413, 'Upload body too large') from exc
        except Exception as exc:
            if request.scope.get('state', {}).get('body_cap_exceeded'):
                raise HTTPException(413, 'Upload body too large') from exc
            raise HTTPException(400, 'Malformed multipart upload') from exc
        root = None
        try:
            if set(form.keys()) != {'files'} or not all(isinstance(x, UploadFile) for x in form.getlist('files')):
                raise HTTPException(422, 'Multipart upload must contain only files')
            files = form.getlist('files')
            if not 1 <= len(files) <= LIMITS['max_files']:
                raise HTTPException(422, 'Upload between 1 and 10 files')
            cid = uuid.uuid4().hex
            root = state.root / sid / cid
            uploads = root / 'uploads'
            uploads.mkdir(parents=True)
            metadata, paths, total = [], [], 0
            for index, upload_file in enumerate(files):
                name = Path(upload_file.filename or '').name
                ext = Path(name).suffix.lower()
                if ext not in EXTENSIONS:
                    raise HTTPException(422, 'Unsupported file type')
                path = uploads / f'{index:02d}{ext}'
                size = 0
                with path.open('wb') as target:
                    while chunk := await upload_file.read(1024 * 1024):
                        size += len(chunk)
                        total += len(chunk)
                        if size > LIMITS['max_file_bytes'] or total > LIMITS['max_batch_bytes']:
                            raise HTTPException(413, 'Upload too large')
                        target.write(chunk)
                if size == 0:
                    raise HTTPException(422, 'Empty upload')
                try:
                    if ext == '.pdf':
                        with path.open('rb') as source:
                            if source.read(5) != b'%PDF-':
                                raise ValueError
                        from starlette.concurrency import run_in_threadpool
                        from .web_jobs import preflight_pdf
                        await run_in_threadpool(preflight_pdf, path)
                    else:
                        from starlette.concurrency import run_in_threadpool
                        def validate_image():
                            from PIL import Image, ImageFile
                            ImageFile.LOAD_TRUNCATED_IMAGES = False
                            with Image.open(path) as image:
                                if image.width * image.height > 100_000_000:
                                    raise ValueError
                                image.load()
                        await run_in_threadpool(validate_image)
                except Exception as exc:
                    raise HTTPException(422, 'Invalid or oversized file') from exc
                metadata.append({'name': name, 'size': size})
                paths.append(str(path))
            record = {'id': cid, 'files': metadata, 'status': 'queued', 'stage': 'queued', 'error': None, 'submitted': clock(), 'started': None, 'finished': None}
            page_names = {Path(p).stem: m['name'] for p, m in zip(paths, metadata)}
            collection = {'record': record, 'root': root, 'questions': {}, 'display': page_names}
            job = {'kind': 'collection', 'root': root, 'progress_path': root / 'progress.json', 'log_path': root / 'job.log', 'payload': {'files': paths},
                   'start': lambda: record.update(status='running', stage='prepare', started=clock()),
                   'finish': lambda _: record.update(status='ready', stage='ready', finished=clock()),
                   'fail': lambda category, diagnostic: record.update(_failure(category, diagnostic), finished=clock())}
            with state.lock:
                info['collections'][cid] = collection
            try:
                state.enqueue(job)
            except Exception:
                with state.lock:
                    info['collections'].pop(cid, None)
                raise
            with state.lock:
                return JSONResponse(_timed(record, root / 'progress.json', 'collection', clock), status_code=202)
        except Exception:
            if root and root.exists():
                import shutil
                shutil.rmtree(root, ignore_errors=True)
            raise
        finally:
            await form.close()

    @app.get('/api/collections/{collection_id}')
    def get_collection(request: Request, collection_id: str):
        _, info = session(request)
        col = info['collections'].get(collection_id)
        if not col:
            raise HTTPException(404, 'Collection not found')
        with state.lock:
            return _timed(col['record'], col['root'] / 'progress.json', 'collection', clock)

    @app.post('/api/collections/{collection_id}/questions')
    async def question(request: Request, collection_id: str, x_csrf_token: str | None = Header(None)):
        sid, info = csrf(request, x_csrf_token)
        col = info['collections'].get(collection_id)
        if not col:
            raise HTTPException(404, 'Collection not found')
        with state.lock:
            if col['record']['status'] != 'ready':
                raise HTTPException(409, 'Collection is not ready')
        try:
            body = await request.json()
            text = body.get('question', '').strip()
        except Exception:
            text = ''
        if not isinstance(text, str) or not text or len(text) > 2000:
            raise HTTPException(422, 'Question must contain 1 to 2000 characters')
        jid = uuid.uuid4().hex
        job_record = {'id': jid, 'question': text, 'status': 'queued', 'stage': 'queued', 'error': None, 'result': None, 'submitted': clock(), 'started': None, 'finished': None}
        job_dir = state.root / sid / collection_id / 'questions' / jid
        image_dir = job_dir / 'images'

        def finish(data):
            images, image_map = [], {}
            image_dir.mkdir(parents=True, exist_ok=True)
            for raw in data.get('images', []):
                path = _contained_file(raw, image_dir)
                image_id = uuid.uuid4().hex
                image_map[image_id] = path
                images.append({'name': path.name, 'url': f'/api/collections/{collection_id}/questions/{jid}/images/{image_id}'})
            evidence = []
            for item in data['evidence']:
                value = dict(item)
                stem = value['doc'].split('-p', 1)[0]
                source = col['display'].get(stem, stem)
                page = re.search(r'-p(\d+)$', value['doc'])
                value['doc'] = f'{source}, page {page.group(1)}' if page else source
                evidence.append(value)
            with state.lock:
                job_record['image_map'] = image_map
                job_record.update(status='ready', stage='ready', finished=clock(), result={k: data[k] for k in ('answer', 'queries')} | {'evidence': evidence, 'images': images})

        def fail(category, diagnostic):
            with state.lock:
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / 'job.log').write_text(str(diagnostic), encoding='utf-8')
                job_record.update(_failure(category, diagnostic), finished=clock())

        def start():
            with state.lock:
                job_record.update(status='running', stage='retrieve', started=clock())

        job = {'kind': 'question', 'root': col['root'], 'progress_path': job_dir / 'progress.json', 'log_path': job_dir / 'job.log',
               'payload': {'question': text, 'session': sid, 'collection': collection_id, 'job': jid},
               'start': start, 'finish': finish, 'fail': fail}
        with state.lock:
            col['questions'][jid] = job_record
        try:
            state.enqueue(job)
        except Exception:
            with state.lock:
                col['questions'].pop(jid, None)
            raise
        with state.lock:
            return JSONResponse(_timed(job_record, job_dir / 'progress.json', 'question', clock), status_code=202)

    @app.get('/api/collections/{collection_id}/questions/{question_id}')
    def get_question(request: Request, collection_id: str, question_id: str):
        _, info = session(request)
        with state.lock:
            job = info['collections'].get(collection_id, {}).get('questions', {}).get(question_id)
            if not job:
                raise HTTPException(404, 'Question not found')
            return _timed(job, state.root / request.cookies.get('docsearch_session', '') / collection_id / 'questions' / question_id / 'progress.json', 'question', clock)

    @app.get('/api/collections/{collection_id}/questions/{question_id}/images/{image_id}')
    def image(request: Request, collection_id: str, question_id: str, image_id: str):
        sid, info = session(request)
        with state.lock:
            job = info['collections'].get(collection_id, {}).get('questions', {}).get(question_id)
            path = job.get('image_map', {}).get(image_id) if job else None
        if not path:
            raise HTTPException(404, 'Image not found')
        exact_job_dir = state.root / sid / collection_id / 'questions' / question_id / 'images'
        try:
            path = _contained_file(path, exact_job_dir)
        except (ValueError, OSError):
            raise HTTPException(404, 'Image not found')
        return FileResponse(path)

    return app
