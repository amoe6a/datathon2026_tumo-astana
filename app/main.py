import asyncio
from contextlib import asynccontextmanager
import hmac
import json
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
import uuid
from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.config import Settings, ROOT
from app.limits import ApiLimits, local_test_request
from app.readiness import ModelReadiness
from app.lifecycle import RuntimeLease, cleanup_job, valid_job_id
from app.processes import start_child, stop_child

class Jobs:
    def __init__(self, settings):
        self.s = settings
        self.s.runtime.mkdir(parents=True, exist_ok=True)
        self.db_path = self.s.runtime / "jobs.sqlite3"
        self.slot = threading.BoundedSemaphore(1)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._active = None
        self._thread = None
        self._reaper = None
        self._pending = {}
        with self.db() as c:
            c.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL, payload TEXT)")

    def recover(self):
        # Called only after obtaining exclusive ownership of this runtime.
        with self.db() as c:
            for jid, payload in c.execute("SELECT id,payload FROM jobs").fetchall():
                obj=json.loads(payload)
                if obj["status"] == "processing":
                    obj.update(status="failed", error="Server restarted during processing; upload again")
                    c.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(obj), jid))
        for folder in self.s.runtime.iterdir():
            if folder.is_dir() and valid_job_id(folder.name):
                self.cleanup(folder.name)

    def cleanup(self, jid):
        count = cleanup_job(self.s.runtime, jid)
        with self._lock:
            if count:
                self._pending[jid] = count
            else:
                self._pending.pop(jid, None)

    def retry_cleanup(self):
        with self._lock:
            ids = list(self._pending)
        for jid in ids:
            self.cleanup(jid)

    def maintenance(self):
        while not self._closing.wait(5):
            self.retry_cleanup()

    def start(self):
        self._reaper = threading.Thread(target=self.maintenance, daemon=True)
        self._reaper.start()

    def maintenance_info(self):
        with self._lock:
            return {'cleanup_pending_files': sum(self._pending.values()),
                    'accepting_jobs': not self._closing.is_set()}

    def launch(self, jid, video):
        with self._lock:
            if self._closing.is_set():
                raise HTTPException(503, 'Server is shutting down; retry after restart')
            self._thread = threading.Thread(target=self.run, args=(jid, video), daemon=True)
            self._thread.start()

    def close(self):
        self._closing.set()
        with self._lock:
            process, thread = self._active, self._thread
        if process is not None:
            stop_child(process)
        if thread is not None:
            thread.join(timeout=10)
        if self._reaper is not None:
            self._reaper.join(timeout=3)
        self.retry_cleanup()

    def db(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def save(self, jid, payload):
        with self.db() as c:
            c.execute("INSERT INTO jobs VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                      (jid, time.time(), json.dumps(payload)))

    def get(self, jid):
        with self.db() as c:
            row=c.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found")
        obj=json.loads(row[0])
        if obj["status"] == "processing":
            progress=self.s.runtime/jid/"progress.json"
            try:
                obj.update(json.loads(progress.read_text(encoding="utf-8")))
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                # Progress is optional; Windows can briefly lock it during replacement.
                pass
        return obj

    def run(self, jid, video):
        folder=video.parent
        result_path=folder/"result.json"
        cmd=["--input", str(video.resolve()), "--model", str(self.s.model.resolve()),
             "--engine", self.s.engine, "--rules", str(self.s.rules.resolve()), "--output", str(result_path.resolve()),
             "--progress", str((folder/"progress.json").resolve())]
        started=time.monotonic()
        process=None
        payload={"job_id":jid,"status":"failed","error":"Local processing failed; no verdict was produced"}
        try:
            process=start_child('job', cmd)
            with self._lock:
                self._active=process
                closing=self._closing.is_set()
            if closing:
                raise RuntimeError('Server stopped during processing; upload again')
            code=process.wait(timeout=self.s.timeout)
            if code or not result_path.exists():
                raise RuntimeError("Local worker exited without a result")
            result=json.loads(result_path.read_text(encoding="utf-8"))
            if not result["ok"]:
                raise RuntimeError(result["error"])
            payload={"job_id":jid, "status":"done", **result["result"]}
            payload["meta"]["wall_time_sec"]=round(time.monotonic()-started,3)
        except subprocess.TimeoutExpired:
            payload={"job_id":jid,"status":"failed","error":"Processing exceeded the time budget; no verdict was produced"}
        except RuntimeError as error:
            payload={"job_id":jid,"status":"failed","error":str(error)[:500]}
        except Exception:
            payload={"job_id":jid,"status":"failed","error":"Local processing failed; no verdict was produced"}
        finally:
            try:
                if process is not None:
                    stop_child(process)
                self.cleanup(jid)
                self.save(jid,payload)
            finally:
                with self._lock:
                    self._active=None
                self.slot.release()

def create_app(settings=None):
    settings=settings or Settings.from_env()
    jobs=Jobs(settings)
    readiness=ModelReadiness(settings)
    lease=RuntimeLease(settings.runtime)

    @asynccontextmanager
    async def lifespan(app):
        lease.acquire()
        try:
            await asyncio.to_thread(jobs.recover)
            await asyncio.to_thread(readiness.check)
            jobs.start()
            yield
        finally:
            try:
                await asyncio.to_thread(jobs.close)
            finally:
                lease.release()

    app=FastAPI(title="Forte Review", version="0.1.0", docs_url="/docs", lifespan=lifespan)
    app.state.jobs=jobs
    app.state.readiness=readiness
    app.add_middleware(ApiLimits,key=settings.api_key,max_bytes=(settings.max_upload_mb+1)*1024*1024,local_test=settings.local_test)

    def authenticate(x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Invalid or missing X-API-Key")

    @app.middleware("http")
    async def headers(request, call_next):
        response=await call_next(request)
        response.headers["Cache-Control"]="no-store"
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Referrer-Policy"]="no-referrer"
        response.headers["Content-Security-Policy"]="default-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/health")
    def health(request: Request):
        check=readiness.snapshot()
        return {"status":"ok", "engine":settings.engine, "model_present":settings.model.exists(),
                "demo_only":settings.engine=="demo", "ready":check['ready'], "readiness":check,
                "local_test": settings.local_test and local_test_request(request.scope),
                **jobs.maintenance_info()}

    @app.get("/ready")
    def ready(request: Request):
        value=health(request)
        return JSONResponse(value,status_code=200 if value['ready'] else 503)

    @app.post("/api/v1/analyze", status_code=202, dependencies=[Depends(authenticate)])
    async def submit(file: UploadFile = File(...)):
        if Path(file.filename or "").suffix.lower() not in {".mov",".mp4"}:
            await file.close()
            raise HTTPException(415, "Upload a MOV or MP4 video")
        check=readiness.snapshot()
        if not check['ready']:
            await file.close()
            raise HTTPException(503, check['message'])
        if not jobs.slot.acquire(blocking=False):
            await file.close()
            raise HTTPException(429, "One video is already processing; retry shortly", headers={"Retry-After":"5"})
        jid=uuid.uuid4().hex[:16]
        folder=settings.runtime/jid
        video=folder/("input"+Path(file.filename).suffix.lower())
        handed_off=False
        try:
            folder.mkdir()
            total=0
            with video.open("wb") as out:
                while chunk := await file.read(1024*1024):
                    total += len(chunk)
                    if total > settings.max_upload_mb*1024*1024:
                        raise HTTPException(413, "Video exceeds upload limit")
                    out.write(chunk)
            if total == 0:
                raise HTTPException(400,"Video is empty")
            jobs.save(jid,{"job_id":jid,"status":"processing","progress":1,"stage":"Starting local analysis"})
            jobs.launch(jid,video)
            handed_off=True
            return {"job_id":jid,"status":"processing"}
        finally:
            try:
                await file.close()
            finally:
                if not handed_off:
                    try:
                        jobs.cleanup(jid)
                    finally:
                        jobs.slot.release()

    @app.get("/api/v1/jobs/{job_id}",dependencies=[Depends(authenticate)])
    def status(job_id: str):
        # Only opaque IDs are accepted. Never interpret a user-supplied path.
        if not valid_job_id(job_id):
            raise HTTPException(404,"Job not found")
        return jobs.get(job_id)

    @app.get("/api/v1/rules",dependencies=[Depends(authenticate)])
    def rules():
        return json.loads(settings.rules.read_text(encoding="utf-8"))

    @app.get("/api/v1/demo/{name}", dependencies=[Depends(authenticate)])
    def demo_video(name: str):
        if settings.engine != "demo" or name not in {"normal", "event", "multiple"}:
            raise HTTPException(404, "Example unavailable")
        path=ROOT/"data/demo"/(name+".mov")
        if not path.exists():
            raise HTTPException(404, "Run synthetic demo setup first")
        return FileResponse(path, media_type="video/quicktime", filename=name+".mov")

    @app.get("/")
    def index():
        return FileResponse(ROOT/"web/index.html")

    app.mount("/assets",StaticFiles(directory=ROOT/"web"),name="assets")
    return app
